from datetime import datetime, date, timedelta

from django.utils import timezone
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAuthenticated, IsAdminOrManager
from apps.employees.models import Employee
from apps.shifts.models import EmployeeShift
from .models import AttendanceLog
from .serializers import (
    AttendanceLogListSerializer,
    AttendanceLogDetailSerializer,
    CheckInSerializer,
    CheckOutSerializer,
)
from .filters import AttendanceLogFilter


class AttendanceViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Attendance management.
    - list / retrieve: view attendance logs (filtered)
    - check-in / check-out: custom actions
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

    # ─── Check-in ────────────────────────────────────────
    @action(detail=False, methods=['post'], url_path='check-in')
    def check_in(self, request):
        serializer = CheckInSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        employee_code = serializer.validated_data['employee_id']

        try:
            employee = Employee.objects.get(employee_id=employee_code, is_active=True)
        except Employee.DoesNotExist:
            return Response({
                'success': False, 'data': None,
                'message': 'Nhân viên không tồn tại hoặc đã bị vô hiệu hóa.',
            }, status=status.HTTP_404_NOT_FOUND)

        today = timezone.localdate()
        now = timezone.now()

        # Prevent duplicate check-in
        log, created = AttendanceLog.objects.get_or_create(
            employee=employee,
            date=today,
            defaults={'check_in': now, 'status': AttendanceLog.Status.PRESENT},
        )
        if not created:
            if log.check_in:
                return Response({
                    'success': False, 'data': None,
                    'message': 'Nhân viên đã check-in hôm nay rồi.',
                }, status=status.HTTP_400_BAD_REQUEST)
            log.check_in = now
            log.save(update_fields=['check_in'])

        # Determine status based on shift + late_threshold
        attendance_status = self._compute_status(employee, now, today)
        log.status = attendance_status
        log.save(update_fields=['status'])

        return Response({
            'success': True,
            'data': AttendanceLogDetailSerializer(log).data,
            'message': 'Check-in thành công.',
        })

    # ─── Check-out ───────────────────────────────────────
    @action(detail=False, methods=['post'], url_path='check-out')
    def check_out(self, request):
        serializer = CheckOutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        employee_code = serializer.validated_data['employee_id']

        try:
            employee = Employee.objects.get(employee_id=employee_code, is_active=True)
        except Employee.DoesNotExist:
            return Response({
                'success': False, 'data': None,
                'message': 'Nhân viên không tồn tại hoặc đã bị vô hiệu hóa.',
            }, status=status.HTTP_404_NOT_FOUND)

        today = timezone.localdate()
        now = timezone.now()

        try:
            log = AttendanceLog.objects.get(employee=employee, date=today)
        except AttendanceLog.DoesNotExist:
            return Response({
                'success': False, 'data': None,
                'message': 'Nhân viên chưa check-in hôm nay.',
            }, status=status.HTTP_400_BAD_REQUEST)

        if log.check_out:
            return Response({
                'success': False, 'data': None,
                'message': 'Nhân viên đã check-out hôm nay rồi.',
            }, status=status.HTTP_400_BAD_REQUEST)

        log.check_out = now
        log.save(update_fields=['check_out'])

        return Response({
            'success': True,
            'data': AttendanceLogDetailSerializer(log).data,
            'message': 'Check-out thành công.',
        })

    # ─── Today ───────────────────────────────────────────
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

    # ─── Helper ──────────────────────────────────────────
    @staticmethod
    def _compute_status(employee, check_in_time, today):
        """
        Compare check-in time with the employee's shift start_time + late_threshold.
        Returns 'present' or 'late'.
        """
        # Get the latest applicable shift for this employee
        emp_shift = (
            EmployeeShift.objects
            .filter(employee=employee, effective_date__lte=today)
            .order_by('-effective_date')
            .select_related('shift')
            .first()
        )
        if not emp_shift:
            return AttendanceLog.Status.PRESENT  # no shift assigned → present

        shift = emp_shift.shift
        # Build a datetime for shift start on today
        shift_start_dt = timezone.make_aware(
            datetime.combine(today, shift.start_time)
        ) if timezone.is_naive(datetime.combine(today, shift.start_time)) else datetime.combine(today, shift.start_time)

        threshold_dt = shift_start_dt + timedelta(minutes=shift.late_threshold)

        if check_in_time > threshold_dt:
            return AttendanceLog.Status.LATE
        return AttendanceLog.Status.PRESENT
