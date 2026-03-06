from rest_framework import serializers
from apps.accounts.serializers import UserSerializer, UserCreateSerializer
from .models import Department, Employee


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

    class Meta:
        model = Employee
        fields = [
            'id', 'user', 'employee_id', 'department',
            'position', 'phone', 'date_joined',
        ]

    def create(self, validated_data):
        user_data = validated_data.pop('user')
        user_serializer = UserCreateSerializer(data=user_data)
        user_serializer.is_valid(raise_exception=True)
        user = user_serializer.save()
        employee = Employee.objects.create(user=user, **validated_data)
        return employee


class EmployeeUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = [
            'department', 'position', 'phone', 'date_joined',
        ]


class FaceEncodingSerializer(serializers.Serializer):
    face_encoding = serializers.JSONField()
