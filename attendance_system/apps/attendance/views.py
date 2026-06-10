import base64
from datetime import datetime, timedelta

from django.db.models import Max
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from apps.accounts.permissions import IsAuthenticated
from apps.employees.models import Employee
from apps.face_recognition.services import FaceRecognitionService
from apps.shifts.models import EmployeeShift
from .filters import AttendanceLogFilter
from .models import AttendanceLog
from .serializers import (
    AttendanceLogDetailSerializer,
    AttendanceLogListSerializer,
    CheckOutSerializer,
)


class AttendanceViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AttendanceLog.objects.select_related(
        'employee',
        'employee__user',
        'employee__department',
        'employee_shift',
        'employee_shift__shift',
    ).all()
    filterset_class = AttendanceLogFilter
    search_fields = ['employee__employee_id', 'employee__user__first_name']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return AttendanceLogDetailSerializer
        return AttendanceLogListSerializer

    def get_permissions(self):
        if self.action == 'check_in':
            return [AllowAny()]
        return [IsAuthenticated()]

    def get_queryset(self):
        queryset = super().get_queryset()
        user = self.request.user
        if user.is_employee:
            queryset = queryset.filter(employee__user=user)
        elif user.is_manager:
            try:
                queryset = queryset.filter(
                    employee__department=user.employee.department,
                )
            except Employee.DoesNotExist:
                queryset = queryset.none()
        return queryset

    @action(detail=False, methods=['post'], url_path='check-in')
    def check_in(self, request):
        image_bytes = self._extract_image_bytes(request)
        if image_bytes is None:
            return Response({
                'success': False,
                'data': None,
                'message': (
                    'Vui lòng gửi ảnh khuôn mặt qua trường image, file hoặc photo.'
                ),
            }, status=status.HTTP_400_BAD_REQUEST)

        service = FaceRecognitionService()
        found_face, employee_pk, confidence = service.process_image(
            image_bytes,
            detect_threshold=0.6,
        )

        if not found_face:
            error_messages = {
                'invalid_image': 'Không thể giải mã ảnh.',
                'no_face_detected': 'Không phát hiện khuôn mặt trong ảnh.',
                'invalid_face_crop': 'Không thể cắt khuôn mặt từ ảnh.',
                'models_unavailable': 'Mô hình nhận diện chưa sẵn sàng.',
            }
            error_code = getattr(service, 'last_error', None)
            return Response({
                'success': False,
                'data': {'error_code': error_code},
                'message': error_messages.get(
                    error_code,
                    'Không phát hiện khuôn mặt trong ảnh.',
                ),
            }, status=status.HTTP_400_BAD_REQUEST)

        if not employee_pk:
            return Response({
                'success': False,
                'data': {
                    'confidence': confidence,
                    'error_code': 'employee_not_recognized',
                },
                'message': 'Khuôn mặt không khớp với nhân viên đã đăng ký.',
            }, status=status.HTTP_200_OK)

        try:
            employee = Employee.objects.get(pk=employee_pk, is_active=True)
        except Employee.DoesNotExist:
            return Response({
                'success': False,
                'data': None,
                'message': 'Nhân viên không tồn tại hoặc đã bị vô hiệu hóa.',
            }, status=status.HTTP_404_NOT_FOUND)

        employee_name = employee.user.get_full_name() or employee.user.username
        log, attendance_action, attendance_error = (
            self._process_attendance_scan(employee)
        )
        if attendance_error:
            return attendance_error

        data = dict(AttendanceLogDetailSerializer(log).data)
        data['confidence'] = confidence
        data['employee_code'] = employee.employee_id
        data['employee_name'] = employee_name
        data['attendance_action'] = attendance_action

        return Response({
            'success': True,
            'data': data,
            'message': (
                'Chấm công ra thành công.'
                if attendance_action == 'check_out'
                else 'Chấm công vào thành công.'
            ),
        })

    @action(detail=False, methods=['post'], url_path='check-out')
    def check_out(self, request):
        serializer = CheckOutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        employee_code = serializer.validated_data['employee_id']

        try:
            employee = Employee.objects.get(
                employee_id=employee_code,
                is_active=True,
            )
        except Employee.DoesNotExist:
            return Response({
                'success': False,
                'data': None,
                'message': 'Nhân viên không tồn tại hoặc đã bị vô hiệu hóa.',
            }, status=status.HTTP_404_NOT_FOUND)

        log = (
            AttendanceLog.objects
            .filter(
                employee=employee,
                date=timezone.localdate(),
                check_in__isnull=False,
                check_out__isnull=True,
            )
            .select_related('employee_shift__shift')
            .order_by('-check_in')
            .first()
        )
        if not log:
            return Response({
                'success': False,
                'data': None,
                'message': 'Nhân viên không có ca nào đang chấm công.',
            }, status=status.HTTP_400_BAD_REQUEST)

        log.check_out = timezone.now()
        log.save(update_fields=['check_out'])
        return Response({
            'success': True,
            'data': AttendanceLogDetailSerializer(log).data,
            'message': 'Chấm công ra thành công.',
        })

    @action(detail=False, methods=['get'], url_path='today')
    def today(self, request):
        queryset = self.get_queryset().filter(date=timezone.localdate())
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = AttendanceLogListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        return Response({
            'success': True,
            'data': AttendanceLogListSerializer(queryset, many=True).data,
            'message': '',
        })

    @action(detail=False, methods=['get'], url_path='smart-office-access')
    def smart_office_access(self, request):
        if request.user.is_admin:
            return Response({
                'success': True,
                'data': {
                    'can_control': True,
                    'checked_in': False,
                    'checked_out': False,
                    'admin_override': True,
                },
                'message': '',
            })

        try:
            employee = request.user.employee
        except Employee.DoesNotExist:
            return Response({
                'success': True,
                'data': {
                    'can_control': False,
                    'checked_in': False,
                    'checked_out': False,
                    'admin_override': False,
                },
                'message': 'Tài khoản chưa được liên kết với nhân viên.',
            })

        logs = AttendanceLog.objects.filter(
            employee=employee,
            date=timezone.localdate(),
        )
        checked_in = logs.filter(check_in__isnull=False).exists()
        can_control = logs.filter(
            check_in__isnull=False,
            check_out__isnull=True,
        ).exists()
        checked_out = checked_in and not can_control

        return Response({
            'success': True,
            'data': {
                'can_control': can_control,
                'checked_in': checked_in,
                'checked_out': checked_out,
                'admin_override': False,
            },
            'message': (
                ''
                if can_control
                else 'Bạn cần chấm công vào một ca đang hoạt động.'
            ),
        })

    def _process_attendance_scan(self, employee):
        today = timezone.localdate()
        now = timezone.now()
        open_log = (
            AttendanceLog.objects
            .filter(
                employee=employee,
                date=today,
                check_in__isnull=False,
                check_out__isnull=True,
            )
            .select_related('employee_shift__shift')
            .order_by('-check_in')
            .first()
        )
        if open_log:
            shift = (
                open_log.employee_shift.shift
                if open_log.employee_shift
                else None
            )
            if shift and timezone.localtime(now).time() < shift.end_time:
                return None, None, Response({
                    'success': False,
                    'data': {
                        'attendance_action': 'already_checked_in',
                        'shift_name': shift.name,
                        'checkout_time': shift.end_time.strftime('%H:%M'),
                    },
                    'message': (
                        f'Nhân viên đã check-in ca {shift.name}. '
                        f'Chỉ có thể checkout từ {shift.end_time:%H:%M}.'
                    ),
                }, status=status.HTTP_400_BAD_REQUEST)

            open_log.check_out = now
            open_log.save(update_fields=['check_out'])
            return open_log, 'check_out', None

        log, error_response = self._create_check_in_log(employee)
        if error_response:
            return None, None, error_response
        return log, 'check_in', None

    def _create_check_in_log(self, employee):
        today = timezone.localdate()
        now = timezone.now()

        open_log = (
            AttendanceLog.objects
            .filter(
                employee=employee,
                date=today,
                check_in__isnull=False,
                check_out__isnull=True,
            )
            .select_related('employee_shift__shift')
            .order_by('-check_in')
            .first()
        )
        if open_log:
            shift_name = (
                open_log.employee_shift.shift.name
                if open_log.employee_shift
                else 'hiện tại'
            )
            return None, Response({
                'success': False,
                'data': None,
                'message': (
                    f'Nhân viên đang trong ca {shift_name} '
                    'và chưa chấm công ra.'
                ),
            }, status=status.HTTP_400_BAD_REQUEST)

        assignment = self._get_next_assignment(employee, today, now)
        if assignment:
            log, created = AttendanceLog.objects.get_or_create(
                employee=employee,
                employee_shift=assignment,
                date=today,
                defaults={
                    'check_in': now,
                    'status': self._compute_status(assignment, now, today),
                },
            )
            if not created:
                return None, Response({
                    'success': False,
                    'data': None,
                    'message': (
                        f'Nhân viên đã chấm công ca '
                        f'{assignment.shift.name} hôm nay.'
                    ),
                }, status=status.HTTP_400_BAD_REQUEST)
            return log, None

        if self._get_active_assignments(employee, today):
            return None, Response({
                'success': False,
                'data': None,
                'message': (
                    'Không còn ca làm việc nào có thể check-in trong hôm nay.'
                ),
            }, status=status.HTTP_400_BAD_REQUEST)

        legacy_log = AttendanceLog.objects.filter(
            employee=employee,
            employee_shift__isnull=True,
            date=today,
        ).first()
        if legacy_log:
            return None, Response({
                'success': False,
                'data': None,
                'message': 'Nhân viên đã chấm công hôm nay.',
            }, status=status.HTTP_400_BAD_REQUEST)

        return AttendanceLog.objects.create(
            employee=employee,
            date=today,
            check_in=now,
            status=AttendanceLog.Status.PRESENT,
        ), None

    @staticmethod
    def _extract_image_bytes(request):
        image_file = (
            request.FILES.get('image')
            or request.FILES.get('file')
            or request.FILES.get('photo')
        )
        if image_file:
            return image_file.read()

        image_data = (
            request.data.get('image')
            or request.data.get('file')
            or request.data.get('photo')
        )
        if not image_data:
            return None
        if hasattr(image_data, 'read'):
            return image_data.read()
        if isinstance(image_data, str):
            try:
                image_data = image_data.strip()
                if ',' in image_data:
                    image_data = image_data.split(',', 1)[1]
                return base64.b64decode(image_data, validate=True)
            except Exception:
                return None
        return None

    @staticmethod
    def _get_active_assignments(employee, today):
        latest_effective_date = (
            EmployeeShift.objects
            .filter(employee=employee, effective_date__lte=today)
            .aggregate(value=Max('effective_date'))
            .get('value')
        )
        if not latest_effective_date:
            return []

        assignments = (
            EmployeeShift.objects
            .filter(
                employee=employee,
                effective_date=latest_effective_date,
            )
            .select_related('shift')
            .order_by('shift__start_time')
        )
        return [
            assignment
            for assignment in assignments
            if today.weekday() in assignment.shift.work_days
        ]

    @classmethod
    def _get_next_assignment(cls, employee, today, check_in_time):
        assignments = cls._get_active_assignments(employee, today)
        completed_assignment_ids = set(
            AttendanceLog.objects
            .filter(
                employee=employee,
                date=today,
                employee_shift__isnull=False,
            )
            .values_list('employee_shift_id', flat=True)
        )
        available = [
            assignment
            for assignment in assignments
            if assignment.id not in completed_assignment_ids
        ]
        if not available:
            return None

        local_time = timezone.localtime(check_in_time).time()
        for assignment in available:
            if local_time <= assignment.shift.end_time:
                return assignment
        return None

    @staticmethod
    def _compute_status(assignment, check_in_time, today):
        shift = assignment.shift
        shift_start = datetime.combine(today, shift.start_time)
        shift_start_dt = (
            timezone.make_aware(shift_start)
            if timezone.is_naive(shift_start)
            else shift_start
        )
        threshold_dt = shift_start_dt + timedelta(
            minutes=shift.late_threshold,
        )
        if check_in_time > threshold_dt:
            return AttendanceLog.Status.LATE
        return AttendanceLog.Status.PRESENT
