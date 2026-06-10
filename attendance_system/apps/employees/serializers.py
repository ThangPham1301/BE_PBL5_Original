from uuid import uuid4

from django.db import transaction
from rest_framework import serializers
from django.utils import timezone
from apps.accounts.serializers import UserSerializer, UserCreateSerializer
from .models import Department, Employee


EMPLOYEE_USER_ROLES = {'manager', 'employee'}


def validate_employee_user_role(user_data):
    role = user_data.get('role')
    if role and role not in EMPLOYEE_USER_ROLES:
        raise serializers.ValidationError({'role': 'Chỉ được chọn vai trò quản lý hoặc nhân viên.'})
    return user_data


# ─── Department ──────────────────────────────────────────
class DepartmentListSerializer(serializers.ModelSerializer):
    manager_name = serializers.SerializerMethodField()
    employee_count = serializers.SerializerMethodField()

    class Meta:
        model = Department
        fields = ['id', 'name', 'manager', 'manager_name', 'employee_count', 'created_at']

    def get_manager_name(self, obj):
        if obj.manager:
            return str(obj.manager)
        return None

    def get_employee_count(self, obj):
        return obj.employees.filter(is_active=True).count()


class DepartmentDetailSerializer(serializers.ModelSerializer):
    manager_name = serializers.SerializerMethodField()

    class Meta:
        model = Department
        fields = ['id', 'name', 'manager', 'manager_name', 'created_at']

    def get_manager_name(self, obj):
        if obj.manager:
            return str(obj.manager)
        return None


# ─── Employee ────────────────────────────────────────────
class EmployeeListSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    department_name = serializers.CharField(source='department.name', read_only=True, default=None)

    class Meta:
        model = Employee
        fields = [
            'id', 'user', 'employee_id', 'department', 'department_name',
            'position', 'phone', 'date_joined', 'is_active',
        ]


class EmployeeDetailSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    department_name = serializers.CharField(source='department.name', read_only=True, default=None)
    face_encoding = serializers.JSONField(read_only=True)

    class Meta:
        model = Employee
        fields = [
            'id', 'user', 'employee_id', 'department', 'department_name',
            'position', 'phone', 'face_encoding', 'date_joined', 'is_active',
        ]


class EmployeeCreateSerializer(serializers.ModelSerializer):
    user = UserCreateSerializer()
    employee_id = serializers.CharField(read_only=True)
    department = serializers.PrimaryKeyRelatedField(
        queryset=Department.objects.all(),
        required=False,
        allow_null=True,
    )
    position = serializers.CharField(required=False, allow_blank=True, default='')
    date_joined = serializers.DateField(required=False, default=timezone.localdate)

    class Meta:
        model = Employee
        fields = [
            'id', 'user', 'employee_id', 'department',
            'position', 'phone', 'date_joined',
        ]

    def validate_user(self, value):
        return validate_employee_user_role(value)

    def create(self, validated_data):
        user_data = validated_data.pop('user')
        validated_data['date_joined'] = timezone.localdate()
        validated_data.pop('employee_id', None)

        with transaction.atomic():
            user_serializer = UserCreateSerializer(data=user_data)
            user_serializer.is_valid(raise_exception=True)
            user = user_serializer.save()
            employee = Employee.objects.create(
                user=user,
                employee_id=f'TMP-{uuid4().hex[:16]}',
                **validated_data,
            )
            prefix = 'MGR' if user.role == 'manager' else 'NV'
            employee.employee_id = f'{prefix}{employee.pk:05d}'
            employee.save(update_fields=['employee_id'])

            if user.role == 'employee':
                from apps.shifts.defaults import get_or_create_default_shift
                from apps.shifts.models import EmployeeShift

                EmployeeShift.objects.create(
                    employee=employee,
                    shift=get_or_create_default_shift(),
                    effective_date=employee.date_joined,
                )
        return employee


class EmployeeUpdateSerializer(serializers.ModelSerializer):
    user = serializers.DictField(required=False, write_only=True)
    department = serializers.PrimaryKeyRelatedField(
        queryset=Department.objects.all(),
        required=False,
        allow_null=True,
    )
    position = serializers.CharField(required=False, allow_blank=True)
    date_joined = serializers.DateField(required=False)

    class Meta:
        model = Employee
        fields = [
            'user', 'department', 'position', 'phone', 'date_joined',
        ]

    def validate_user(self, value):
        return validate_employee_user_role(value)

    def update(self, instance, validated_data):
        user_data = validated_data.pop('user', None)
        employee = super().update(instance, validated_data)

        if user_data:
            user = employee.user
            for field in ['username', 'email', 'first_name', 'last_name', 'role']:
                if field in user_data:
                    setattr(user, field, user_data[field])
            password = user_data.get('password')
            if password:
                user.set_password(password)
            user.save()

        return employee


class FaceEncodingSerializer(serializers.Serializer):
    face_encoding = serializers.JSONField()
