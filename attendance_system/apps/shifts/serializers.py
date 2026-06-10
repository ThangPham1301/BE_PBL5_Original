from rest_framework import serializers
from .models import Shift, EmployeeShift
from apps.employees.serializers import EmployeeListSerializer


class ShiftSerializer(serializers.ModelSerializer):
    employees = serializers.SerializerMethodField()
    check_in_time = serializers.TimeField(
        source='start_time',
        format='%H:%M:%S',
        read_only=True,
    )
    check_out_time = serializers.TimeField(
        source='end_time',
        format='%H:%M:%S',
        read_only=True,
    )
    start_time = serializers.TimeField(
        format='%H:%M:%S',
        input_formats=['%H:%M:%S'],
    )
    end_time = serializers.TimeField(
        format='%H:%M:%S',
        input_formats=['%H:%M:%S'],
    )

    class Meta:
        model = Shift
        fields = [
            'id',
            'name',
            'start_time',
            'end_time',
            'check_in_time',
            'check_out_time',
            'work_days',
            'late_threshold',
            'employees',
        ]

    def get_employees(self, obj):
        """Get all employees assigned to this shift."""
        employee_shifts = obj.employee_shifts.all()
        employees = [es.employee for es in employee_shifts]
        return EmployeeListSerializer(employees, many=True).data

    def validate(self, attrs):
        start_time = attrs.get(
            'start_time',
            getattr(self.instance, 'start_time', None),
        )
        end_time = attrs.get(
            'end_time',
            getattr(self.instance, 'end_time', None),
        )
        if start_time and end_time and start_time >= end_time:
            raise serializers.ValidationError({
                'end_time': 'Giờ kết thúc phải sau giờ bắt đầu.',
            })
        return attrs


class MyShiftSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(source='shift.id', read_only=True)
    name = serializers.CharField(source='shift.name', read_only=True)
    start_time = serializers.TimeField(
        source='shift.start_time',
        format='%H:%M:%S',
        read_only=True,
    )
    end_time = serializers.TimeField(
        source='shift.end_time',
        format='%H:%M:%S',
        read_only=True,
    )
    check_in_time = serializers.TimeField(
        source='shift.start_time',
        format='%H:%M:%S',
        read_only=True,
    )
    check_out_time = serializers.TimeField(
        source='shift.end_time',
        format='%H:%M:%S',
        read_only=True,
    )
    work_days = serializers.JSONField(source='shift.work_days', read_only=True)
    late_threshold = serializers.IntegerField(
        source='shift.late_threshold',
        read_only=True,
    )

    class Meta:
        model = EmployeeShift
        fields = [
            'id',
            'name',
            'start_time',
            'end_time',
            'check_in_time',
            'check_out_time',
            'work_days',
            'late_threshold',
            'effective_date',
        ]


class EmployeeShiftSerializer(serializers.ModelSerializer):
    shift_name = serializers.CharField(source='shift.name', read_only=True)
    employee_code = serializers.CharField(
        source='employee.employee_id',
        read_only=True,
    )
    start_time = serializers.TimeField(
        source='shift.start_time',
        format='%H:%M:%S',
        read_only=True,
    )
    end_time = serializers.TimeField(
        source='shift.end_time',
        format='%H:%M:%S',
        read_only=True,
    )
    check_in_time = serializers.TimeField(
        source='shift.start_time',
        format='%H:%M:%S',
        read_only=True,
    )
    check_out_time = serializers.TimeField(
        source='shift.end_time',
        format='%H:%M:%S',
        read_only=True,
    )
    work_days = serializers.JSONField(source='shift.work_days', read_only=True)
    employee_name = serializers.SerializerMethodField()

    class Meta:
        model = EmployeeShift
        fields = [
            'id',
            'employee',
            'employee_code',
            'shift',
            'shift_name',
            'start_time',
            'end_time',
            'check_in_time',
            'check_out_time',
            'work_days',
            'employee_name',
            'effective_date',
        ]

    def get_employee_name(self, obj):
        return str(obj.employee)


class AssignShiftSerializer(serializers.Serializer):
    employee_id = serializers.IntegerField()
    shift_id = serializers.IntegerField()
    effective_date = serializers.DateField()
