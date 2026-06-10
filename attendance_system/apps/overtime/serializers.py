from django.utils import timezone
from rest_framework import serializers

from apps.employees.models import Employee

from .models import OvertimeRequest


class OvertimeRequestSerializer(serializers.ModelSerializer):
    employee_name = serializers.SerializerMethodField()
    employee_code = serializers.CharField(
        source='employee.employee_id',
        read_only=True,
    )
    department_name = serializers.CharField(
        source='employee.department.name',
        read_only=True,
        allow_null=True,
    )
    reviewed_by_name = serializers.SerializerMethodField()

    class Meta:
        model = OvertimeRequest
        fields = [
            'id',
            'employee',
            'employee_code',
            'employee_name',
            'department_name',
            'date',
            'planned_start_time',
            'planned_end_time',
            'planned_hours',
            'overtime_rate',
            'reason',
            'status',
            'reviewed_by',
            'reviewed_by_name',
            'rejection_reason',
            'reviewed_at',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'employee',
            'planned_hours',
            'overtime_rate',
            'status',
            'reviewed_by',
            'rejection_reason',
            'reviewed_at',
            'created_at',
            'updated_at',
        ]

    def get_employee_name(self, obj):
        return obj.employee.user.get_full_name() or obj.employee.user.username

    def get_reviewed_by_name(self, obj):
        if not obj.reviewed_by:
            return None
        return obj.reviewed_by.get_full_name() or obj.reviewed_by.username

    def validate_reason(self, value):
        return value.strip()

    def validate_date(self, value):
        if value < timezone.localdate():
            raise serializers.ValidationError(
                'Ngày tăng ca phải từ ngày hiện tại trở đi.',
            )
        return value

    def validate(self, attrs):
        start_time = attrs.get('planned_start_time')
        end_time = attrs.get('planned_end_time')
        overtime_date = attrs.get('date')
        if not start_time or not end_time or not overtime_date:
            return attrs

        duration = OvertimeRequest.calculate_hours(start_time, end_time)
        if duration <= 0:
            raise serializers.ValidationError(
                'Giờ kết thúc phải sau giờ bắt đầu.',
            )
        if duration > 2:
            raise serializers.ValidationError(
                'Tăng ca không được quá 2 tiếng mỗi ngày.',
            )

        try:
            employee = self.context['request'].user.employee
        except Employee.DoesNotExist as exc:
            raise serializers.ValidationError(
                'Tài khoản chưa được liên kết với nhân viên.',
            ) from exc
        active_requests = OvertimeRequest.objects.filter(
            employee=employee,
            date=overtime_date,
            status__in=[
                OvertimeRequest.Status.PENDING,
                OvertimeRequest.Status.APPROVED,
            ],
        )
        if self.instance:
            active_requests = active_requests.exclude(pk=self.instance.pk)
        if active_requests.filter(
            planned_start_time__lt=end_time,
            planned_end_time__gt=start_time,
        ).exists():
            raise serializers.ValidationError(
                'Khoảng thời gian tăng ca bị trùng với một đơn đang chờ hoặc đã duyệt.',
            )
        total_hours = duration + sum(
            (
                OvertimeRequest.calculate_hours(
                    request.planned_start_time,
                    request.planned_end_time,
                )
                for request in active_requests
            ),
            start=0,
        )
        if total_hours > 2:
            raise serializers.ValidationError(
                'Tổng thời gian tăng ca không được quá 2 tiếng mỗi ngày.',
            )
        return attrs
