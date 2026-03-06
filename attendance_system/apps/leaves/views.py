from datetime import timedelta

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAuthenticated, IsAdminOrManager
from apps.attendance.models import AttendanceLog
from apps.employees.models import Employee
from .models import LeaveType, LeaveRequest
from .serializers import (
    LeaveTypeSerializer,
    LeaveRequestListSerializer,
    LeaveRequestCreateSerializer,
)


class LeaveTypeViewSet(viewsets.ModelViewSet):
    queryset = LeaveType.objects.all().order_by('name')
    serializer_class = LeaveTypeSerializer

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            from apps.accounts.permissions import IsAdmin
            return [IsAdmin()]
        return [IsAuthenticated()]


class LeaveRequestViewSet(viewsets.ModelViewSet):
    queryset = LeaveRequest.objects.select_related(
        'employee', 'employee__user', 'leave_type', 'approved_by'
    ).all()
    http_method_names = ['get', 'post', 'put', 'patch', 'head', 'options']

    def get_serializer_class(self):
        if self.action == 'create':
            return LeaveRequestCreateSerializer
        return LeaveRequestListSerializer

    def get_permissions(self):
        if self.action in ['approve', 'reject']:
            return [IsAdminOrManager()]
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

    def perform_create(self, serializer):
        """Employee creates a leave request for themselves."""
        try:
            employee = self.request.user.employee
        except Employee.DoesNotExist:
            from rest_framework.exceptions import ValidationError
            raise ValidationError('Tài khoản chưa được liên kết với nhân viên.')
        serializer.save(employee=employee)

    @action(detail=True, methods=['put'], url_path='approve')
    def approve(self, request, pk=None):
        leave_request = self.get_object()
        if leave_request.status != LeaveRequest.Status.PENDING:
            return Response({
                'success': False, 'data': None,
                'message': 'Chỉ có thể duyệt đơn đang ở trạng thái chờ.',
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            approver = request.user.employee
        except Employee.DoesNotExist:
            approver = None

        leave_request.status = LeaveRequest.Status.APPROVED
        leave_request.approved_by = approver
        leave_request.save(update_fields=['status', 'approved_by'])

        # Create AttendanceLog with status='leave' for each leave day
        self._create_leave_attendance_logs(leave_request)

        return Response({
            'success': True,
            'data': LeaveRequestListSerializer(leave_request).data,
            'message': 'Đã duyệt đơn nghỉ phép.',
        })

    @action(detail=True, methods=['put'], url_path='reject')
    def reject(self, request, pk=None):
        leave_request = self.get_object()
        if leave_request.status != LeaveRequest.Status.PENDING:
            return Response({
                'success': False, 'data': None,
                'message': 'Chỉ có thể từ chối đơn đang ở trạng thái chờ.',
            }, status=status.HTTP_400_BAD_REQUEST)

        leave_request.status = LeaveRequest.Status.REJECTED
        leave_request.save(update_fields=['status'])

        return Response({
            'success': True,
            'data': LeaveRequestListSerializer(leave_request).data,
            'message': 'Đã từ chối đơn nghỉ phép.',
        })

    @staticmethod
    def _create_leave_attendance_logs(leave_request):
        """Create attendance logs with status=leave for each day in the leave period."""
        current = leave_request.start_date
        while current <= leave_request.end_date:
            AttendanceLog.objects.update_or_create(
                employee=leave_request.employee,
                date=current,
                defaults={
                    'status': AttendanceLog.Status.LEAVE,
                    'note': f'Nghỉ phép: {leave_request.leave_type.name}',
                },
            )
            current += timedelta(days=1)
