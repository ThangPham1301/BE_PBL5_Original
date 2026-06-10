from django.db import transaction
from django.db.models import Max
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAdmin, IsAdminOrManager, IsAuthenticated
from apps.employees.models import Employee
from .models import EmployeeShift, Shift
from .serializers import (
    AssignShiftSerializer,
    EmployeeShiftSerializer,
    MyShiftSerializer,
    ShiftSerializer,
)


class ShiftViewSet(viewsets.ModelViewSet):
    queryset = Shift.objects.all().order_by('start_time')
    serializer_class = ShiftSerializer

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [IsAdmin()]
        if self.action in ['assign', 'assignments', 'unassign']:
            return [IsAdminOrManager()]
        return [IsAuthenticated()]

    @action(detail=False, methods=['get'], url_path='my')
    def my_shift(self, request):
        try:
            employee = request.user.employee
        except Employee.DoesNotExist:
            return Response({
                'success': False,
                'data': None,
                'message': 'Tài khoản chưa được liên kết với nhân viên.',
            }, status=status.HTTP_404_NOT_FOUND)

        latest_effective_date = (
            EmployeeShift.objects
            .filter(employee=employee, effective_date__lte=timezone.localdate())
            .aggregate(value=Max('effective_date'))
            .get('value')
        )
        if not latest_effective_date:
            return Response({
                'success': True,
                'data': [],
                'message': 'Nhân viên chưa được gán ca làm việc.',
            })

        assignments = (
            EmployeeShift.objects
            .filter(
                employee=employee,
                effective_date=latest_effective_date,
            )
            .select_related('shift')
            .order_by('shift__start_time', 'shift__end_time')
        )
        return Response({
            'success': True,
            'data': MyShiftSerializer(assignments, many=True).data,
            'message': '',
        })

    @action(detail=False, methods=['post'], url_path='assign')
    def assign(self, request):
        serializer = AssignShiftSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        with transaction.atomic():
            employee_queryset = Employee.objects.select_for_update().filter(
                pk=data['employee_id'],
                is_active=True,
            )
            if request.user.is_manager:
                try:
                    employee_queryset = employee_queryset.filter(
                        department=request.user.employee.department,
                    )
                except Employee.DoesNotExist:
                    employee_queryset = employee_queryset.none()

            employee = employee_queryset.first()
            if not employee:
                return Response({
                    'success': False,
                    'data': None,
                    'message': (
                        'Nhân viên không tồn tại hoặc không thuộc phạm vi quản lý.'
                    ),
                }, status=status.HTTP_404_NOT_FOUND)

            try:
                shift = Shift.objects.get(pk=data['shift_id'])
            except Shift.DoesNotExist:
                return Response({
                    'success': False,
                    'data': None,
                    'message': 'Ca làm việc không tồn tại.',
                }, status=status.HTTP_404_NOT_FOUND)

            assignments = list(
                EmployeeShift.objects
                .filter(
                    employee=employee,
                    effective_date=data['effective_date'],
                )
                .select_related('shift')
            )
            inherited_assignments = []
            if not assignments:
                previous_effective_date = (
                    EmployeeShift.objects
                    .filter(
                        employee=employee,
                        effective_date__lt=data['effective_date'],
                    )
                    .aggregate(value=Max('effective_date'))
                    .get('value')
                )
                if previous_effective_date:
                    inherited_assignments = list(
                        EmployeeShift.objects
                        .filter(
                            employee=employee,
                            effective_date=previous_effective_date,
                        )
                        .select_related('shift')
                    )
                    assignments = inherited_assignments

            for assignment in assignments:
                assigned_shift = assignment.shift
                same_work_day = bool(
                    set(shift.work_days) & set(assigned_shift.work_days)
                )
                overlaps = (
                    shift.start_time < assigned_shift.end_time
                    and shift.end_time > assigned_shift.start_time
                )
                if same_work_day and overlaps:
                    return Response({
                        'success': False,
                        'data': {
                            'conflicting_assignment_id': assignment.id,
                            'conflicting_shift': assigned_shift.name,
                        },
                        'message': (
                            f'Ca {shift.name} '
                            f'({shift.start_time:%H:%M}-{shift.end_time:%H:%M}) '
                            f'bị cấn giờ với ca {assigned_shift.name} '
                            f'({assigned_shift.start_time:%H:%M}-'
                            f'{assigned_shift.end_time:%H:%M}).'
                        ),
                    }, status=status.HTTP_400_BAD_REQUEST)

            if inherited_assignments:
                EmployeeShift.objects.bulk_create([
                    EmployeeShift(
                        employee=employee,
                        shift_id=assignment.shift_id,
                        effective_date=data['effective_date'],
                    )
                    for assignment in inherited_assignments
                ])

            employee_shift = EmployeeShift.objects.create(
                employee=employee,
                shift=shift,
                effective_date=data['effective_date'],
            )

        return Response({
            'success': True,
            'data': EmployeeShiftSerializer(employee_shift).data,
            'message': 'Đã gán ca làm việc cho nhân viên.',
        }, status=status.HTTP_201_CREATED)

    def _get_manageable_assignments(self, request):
        assignments = EmployeeShift.objects.select_related(
            'employee',
            'employee__user',
            'employee__department',
            'shift',
        )
        if request.user.is_manager:
            try:
                assignments = assignments.filter(
                    employee__department=request.user.employee.department,
                )
            except Employee.DoesNotExist:
                return assignments.none()
        return assignments

    @action(detail=False, methods=['get'], url_path='assignments')
    def assignments(self, request):
        assignments = self._get_manageable_assignments(request)
        employee_id = request.query_params.get('employee_id')
        if employee_id:
            assignments = assignments.filter(employee_id=employee_id)

        assignments = assignments.order_by(
            'employee__employee_id',
            '-effective_date',
            'shift__start_time',
        )
        return Response({
            'success': True,
            'data': EmployeeShiftSerializer(assignments, many=True).data,
            'message': '',
        })

    @action(
        detail=False,
        methods=['delete'],
        url_path=r'assignments/(?P<assignment_id>\d+)',
    )
    def unassign(self, request, assignment_id=None):
        assignment = (
            self._get_manageable_assignments(request)
            .filter(pk=assignment_id)
            .first()
        )
        if not assignment:
            return Response({
                'success': False,
                'data': None,
                'message': (
                    'Phân ca không tồn tại hoặc không thuộc phạm vi quản lý.'
                ),
            }, status=status.HTTP_404_NOT_FOUND)

        if assignment.attendance_logs.exists():
            return Response({
                'success': False,
                'data': None,
                'message': (
                    'Không thể xóa ca đã có dữ liệu chấm công. '
                    'Hãy giữ lại để bảo toàn lịch sử.'
                ),
            }, status=status.HTTP_400_BAD_REQUEST)

        assignment.delete()
        return Response({
            'success': True,
            'data': None,
            'message': 'Đã xóa ca làm khỏi nhân viên.',
        })
