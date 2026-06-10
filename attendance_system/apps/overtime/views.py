from django.db import transaction
from django.utils import timezone
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response

from apps.accounts.permissions import (
    IsAdminOrManager,
    IsAuthenticated,
    IsEmployee,
)
from apps.employees.models import Employee

from .models import OvertimeRequest
from .serializers import OvertimeRequestSerializer


class OvertimeRequestViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    serializer_class = OvertimeRequestSerializer
    http_method_names = ['get', 'post', 'put', 'head', 'options']
    filterset_fields = ['status', 'date', 'employee']
    ordering_fields = ['date', 'created_at', 'planned_start_time']
    search_fields = [
        'employee__employee_id',
        'employee__user__first_name',
        'employee__user__last_name',
        'reason',
    ]

    def get_permissions(self):
        if self.action == 'create':
            return [IsEmployee()]
        if self.action in ['approve', 'reject']:
            return [IsAdminOrManager()]
        return [IsAuthenticated()]

    def get_queryset(self):
        queryset = OvertimeRequest.objects.select_related(
            'employee',
            'employee__user',
            'employee__department',
            'reviewed_by',
        )
        user = self.request.user
        if user.is_employee:
            return queryset.filter(employee__user=user)
        if user.is_manager:
            try:
                department_id = user.employee.department_id
            except Employee.DoesNotExist:
                return queryset.none()
            if not department_id:
                return queryset.none()
            return queryset.filter(employee__department_id=department_id)
        if user.is_admin:
            return queryset
        return queryset.none()

    def perform_create(self, serializer):
        try:
            employee = self.request.user.employee
        except Employee.DoesNotExist as exc:
            raise ValidationError(
                'Tài khoản chưa được liên kết với nhân viên.',
            ) from exc
        serializer.save(employee=employee)

    @action(detail=True, methods=['put'], url_path='approve')
    def approve(self, request, pk=None):
        with transaction.atomic():
            overtime_request = self._get_pending_request(pk)
            if isinstance(overtime_request, Response):
                return overtime_request

            overtime_request.status = OvertimeRequest.Status.APPROVED
            overtime_request.reviewed_by = request.user
            overtime_request.reviewed_at = timezone.now()
            overtime_request.rejection_reason = ''
            overtime_request.save(update_fields=[
                'status',
                'reviewed_by',
                'reviewed_at',
                'rejection_reason',
                'updated_at',
            ])
        return Response({
            'success': True,
            'data': self.get_serializer(overtime_request).data,
            'message': 'Đã duyệt yêu cầu tăng ca.',
        })

    @action(detail=True, methods=['put'], url_path='reject')
    def reject(self, request, pk=None):
        with transaction.atomic():
            overtime_request = self._get_pending_request(pk)
            if isinstance(overtime_request, Response):
                return overtime_request

            rejection_reason = str(request.data.get('reason', '')).strip()
            if not rejection_reason:
                rejection_reason = 'Từ chối bởi quản lý'

            overtime_request.status = OvertimeRequest.Status.REJECTED
            overtime_request.reviewed_by = request.user
            overtime_request.reviewed_at = timezone.now()
            overtime_request.rejection_reason = rejection_reason
            overtime_request.save(update_fields=[
                'status',
                'reviewed_by',
                'reviewed_at',
                'rejection_reason',
                'updated_at',
            ])
        return Response({
            'success': True,
            'data': self.get_serializer(overtime_request).data,
            'message': 'Đã từ chối yêu cầu tăng ca.',
        })

    def _get_pending_request(self, pk):
        overtime_request = (
            self.get_queryset()
            .select_for_update()
            .filter(pk=pk)
            .first()
        )
        if not overtime_request:
            return Response({
                'success': False,
                'data': None,
                'message': (
                    'Đơn tăng ca không tồn tại hoặc không thuộc phạm vi quản lý.'
                ),
            }, status=status.HTTP_404_NOT_FOUND)
        if overtime_request.status != OvertimeRequest.Status.PENDING:
            return Response({
                'success': False,
                'data': None,
                'message': 'Chỉ có thể xử lý đơn đang chờ duyệt.',
            }, status=status.HTTP_400_BAD_REQUEST)
        return overtime_request
