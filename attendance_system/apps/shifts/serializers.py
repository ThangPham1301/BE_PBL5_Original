from rest_framework import serializers
from .models import Shift, EmployeeShift
from apps.employees.serializers import EmployeeListSerializer


class ShiftSerializer(serializers.ModelSerializer):
    employees = serializers.SerializerMethodField()

    class Meta:
        model = Shift
        fields = ['id', 'name', 'start_time', 'end_time', 'work_days', 'late_threshold', 'employees']

    def get_employees(self, obj):
        """Get all employees assigned to this shift."""
        employee_shifts = obj.employee_shifts.all()
        employees = [es.employee for es in employee_shifts]
        return EmployeeListSerializer(employees, many=True).data


class MyShiftSerializer(serializers.ModelSerializer):
    class Meta:
        model = Shift
        fields = ['id', 'name', 'start_time', 'end_time', 'work_days', 'late_threshold']


class EmployeeShiftSerializer(serializers.ModelSerializer):
    shift_name = serializers.CharField(source='shift.name', read_only=True)
    employee_name = serializers.SerializerMethodField()

    class Meta:
        model = EmployeeShift
        fields = ['id', 'employee', 'shift', 'shift_name', 'employee_name', 'effective_date']

    def get_employee_name(self, obj):
        return str(obj.employee)


class AssignShiftSerializer(serializers.Serializer):
    employee_id = serializers.IntegerField()
    shift_id = serializers.IntegerField()
    effective_date = serializers.DateField()
