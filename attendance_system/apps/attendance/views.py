import base64
from datetime import datetime, timedelta

from django.utils import timezone
from rest_framework import viewsets, status
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
    """
    Attendance management.
    - list / retrieve: view attendance logs
    - check-in: public face check-in by image
    - check-out: check-out by employee code
    - today: attendance for today
    """
    queryset = AttendanceLog.objects.select_related(
        'employee', 'employee__user', 'employee__department'
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
        qs = super().get_queryset()
        user = self.request.user
        if user.is_employee:
            qs = qs.filter(employee__user=user)
        elif user.is_manager:
            try:
                manager_employee = user.employee
                qs = qs.filter(employee__department=manager_employee.department)
            except Employee.DoesNotExist:
                qs = qs.none()
        return qs

    @action(detail=False, methods=['post'], url_path='check-in')
    def check_in(self, request):
        image_bytes = self._extract_image_bytes(request)
        if image_bytes is None:
            return Response({
                'success': False,
                'data': None,
                'message': 'Vui long gui anh khuon mat qua field image, file hoac photo.',
            }, status=status.HTTP_400_BAD_REQUEST)

        service = FaceRecognitionService()
        found_face, employee_pk, confidence = service.process_image(image_bytes, detect_threshold=0.6)

        if not found_face:
            error_messages = {
                'invalid_image': 'Anh gui len khong decode duoc. Hay gui file jpg/png hoac base64 anh hop le.',
                'no_face_detected': 'Khong phat hien khuon mat trong anh. Hay chup ro mat, du sang va gan camera hon.',
                'invalid_face_crop': 'Khong crop duoc khuon mat tu anh. Hay thu lai voi anh ro hon.',
                'models_unavailable': 'Model nhan dien khuon mat chua san sang.',
            }
            error_code = getattr(service, 'last_error', None)
            return Response({
                'success': False,
                'data': {'error_code': error_code},
                'message': error_messages.get(error_code, 'Khong phat hien khuon mat trong anh.'),
            }, status=status.HTTP_400_BAD_REQUEST)

        if not employee_pk:
            return Response({
                'success': False,
                'data': {
                    'confidence': confidence,
                    'error_code': 'employee_not_recognized',
                },
                'message': 'Co khuon mat nhung khong khop voi nhan vien nao trong du lieu dang ky.',
            }, status=status.HTTP_200_OK)

        try:
            employee = Employee.objects.get(pk=employee_pk, is_active=True)
        except Employee.DoesNotExist:
            return Response({
                'success': False,
                'data': None,
                'message': 'Nhan vien khong ton tai hoac da bi vo hieu hoa.',
            }, status=status.HTTP_404_NOT_FOUND)

        employee_name = employee.user.get_full_name() or employee.user.username
        print(
            f"[ATTENDANCE CHECK-IN] employee_code={employee.employee_id} "
            f"employee_name={employee_name} confidence={confidence:.4f}"
        )

        log, duplicate_response = self._create_check_in_log(employee)
        if duplicate_response:
            return duplicate_response

        data = dict(AttendanceLogDetailSerializer(log).data)
        data['confidence'] = confidence
        data['employee_code'] = employee.employee_id
        data['employee_name'] = employee_name

        return Response({
            'success': True,
            'data': data,
            'message': 'Check-in thanh cong.',
        })

    @action(detail=False, methods=['post'], url_path='check-out')
    def check_out(self, request):
        serializer = CheckOutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        employee_code = serializer.validated_data['employee_id']

        try:
            employee = Employee.objects.get(employee_id=employee_code, is_active=True)
        except Employee.DoesNotExist:
            return Response({
                'success': False,
                'data': None,
                'message': 'Nhan vien khong ton tai hoac da bi vo hieu hoa.',
            }, status=status.HTTP_404_NOT_FOUND)

        today = timezone.localdate()
        now = timezone.now()

        try:
            log = AttendanceLog.objects.get(employee=employee, date=today)
        except AttendanceLog.DoesNotExist:
            return Response({
                'success': False,
                'data': None,
                'message': 'Nhan vien chua check-in hom nay.',
            }, status=status.HTTP_400_BAD_REQUEST)

        if log.check_out:
            return Response({
                'success': False,
                'data': None,
                'message': 'Nhan vien da check-out hom nay roi.',
            }, status=status.HTTP_400_BAD_REQUEST)

        log.check_out = now
        log.save(update_fields=['check_out'])

        return Response({
            'success': True,
            'data': AttendanceLogDetailSerializer(log).data,
            'message': 'Check-out thanh cong.',
        })

    @action(detail=False, methods=['get'], url_path='today')
    def today(self, request):
        today = timezone.localdate()
        qs = self.get_queryset().filter(date=today)
        page = self.paginate_queryset(qs)
        if page is not None:
            serializer = AttendanceLogListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = AttendanceLogListSerializer(qs, many=True)
        return Response({
            'success': True,
            'data': serializer.data,
            'message': '',
        })

    def _create_check_in_log(self, employee):
        today = timezone.localdate()
        now = timezone.now()

        log, created = AttendanceLog.objects.get_or_create(
            employee=employee,
            date=today,
            defaults={'check_in': now, 'status': AttendanceLog.Status.PRESENT},
        )

        if not created:
            if log.check_in:
                return None, Response({
                    'success': False,
                    'data': None,
                    'message': 'Nhan vien da check-in hom nay roi.',
                }, status=status.HTTP_400_BAD_REQUEST)
            log.check_in = now
            log.save(update_fields=['check_in'])

        attendance_status = self._compute_status(employee, now, today)
        log.status = attendance_status
        log.save(update_fields=['status'])

        return log, None

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
    def _compute_status(employee, check_in_time, today):
        emp_shift = (
            EmployeeShift.objects
            .filter(employee=employee, effective_date__lte=today)
            .order_by('-effective_date')
            .select_related('shift')
            .first()
        )
        if not emp_shift:
            return AttendanceLog.Status.PRESENT

        shift = emp_shift.shift
        shift_start = datetime.combine(today, shift.start_time)
        shift_start_dt = (
            timezone.make_aware(shift_start)
            if timezone.is_naive(shift_start)
            else shift_start
        )
        threshold_dt = shift_start_dt + timedelta(minutes=shift.late_threshold)

        if check_in_time > threshold_dt:
            return AttendanceLog.Status.LATE
        return AttendanceLog.Status.PRESENT
