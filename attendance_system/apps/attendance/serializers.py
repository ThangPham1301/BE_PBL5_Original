from rest_framework import serializers
from .models import AttendanceLog


class AttendanceLogListSerializer(serializers.ModelSerializer):
    employee_name = serializers.SerializerMethodField()
    employee_code = serializers.CharField(source='employee.employee_id', read_only=True)
    department_name = serializers.CharField(source='employee.department.name', read_only=True, default=None)

    class Meta:
        model = AttendanceLog
        fields = [
            'id', 'employee', 'employee_name', 'employee_code',
            'department_name', 'date', 'check_in', 'check_out',
            'status', 'note', 'created_at',
        ]

    def get_employee_name(self, obj):
        return obj.employee.user.get_full_name() or obj.employee.user.username


class AttendanceLogDetailSerializer(AttendanceLogListSerializer):
    class Meta(AttendanceLogListSerializer.Meta):
        fields = AttendanceLogListSerializer.Meta.fields


class CheckInSerializer(serializers.Serializer):
    employee_id = serializers.CharField(help_text='Mã nhân viên')
    face_encoding = serializers.JSONField(required=False, help_text='Dữ liệu khuôn mặt')


class CheckOutSerializer(serializers.Serializer):
    employee_id = serializers.CharField(help_text='Mã nhân viên')
