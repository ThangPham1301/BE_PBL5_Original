from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAdmin, IsAdminOrManager, IsAuthenticated
from .models import Department, Employee
from .serializers import (
    DepartmentListSerializer,
    DepartmentDetailSerializer,
    EmployeeListSerializer,
    EmployeeDetailSerializer,
    EmployeeCreateSerializer,
    EmployeeUpdateSerializer,
    FaceEncodingSerializer,
)
from .filters import EmployeeFilter


class DepartmentViewSet(viewsets.ModelViewSet):
    queryset = Department.objects.all().order_by('name')
    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'list':
            return DepartmentListSerializer
        return DepartmentDetailSerializer

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [IsAdmin()]
        return [IsAuthenticated()]

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.employees.filter(is_active=True).exists():
            return Response({
                'success': False,
                'data': None,
                'message': 'Không thể xóa phòng ban đang có nhân viên hoạt động.',
            }, status=status.HTTP_400_BAD_REQUEST)
        return super().destroy(request, *args, **kwargs)


class EmployeeViewSet(viewsets.ModelViewSet):
    queryset = Employee.objects.select_related('user', 'department').all().order_by('employee_id')
    filterset_class = EmployeeFilter
    search_fields = ['employee_id', 'user__first_name', 'user__last_name']

    def get_serializer_class(self):
        if self.action == 'create':
            return EmployeeCreateSerializer
        if self.action in ['update', 'partial_update']:
            return EmployeeUpdateSerializer
        if self.action == 'retrieve':
            return EmployeeDetailSerializer
        return EmployeeListSerializer

    def get_permissions(self):
        if self.action in ['create', 'destroy']:
            return [IsAdmin()]
        if self.action in ['update', 'partial_update']:
            return [IsAdminOrManager()]
        return [IsAuthenticated()]

    def get_queryset(self):
        qs = super().get_queryset()
        user = self.request.user
        if user.is_employee:
            qs = qs.filter(user=user)
        elif user.is_manager:
            # Manager sees employees in their department
            try:
                manager_employee = user.employee
                qs = qs.filter(department=manager_employee.department)
            except Employee.DoesNotExist:
                qs = qs.none()
        return qs

    def destroy(self, request, *args, **kwargs):
        """Soft delete: set is_active = False."""
        instance = self.get_object()
        instance.is_active = False
        instance.save(update_fields=['is_active'])
        instance.user.is_active = False
        instance.user.save(update_fields=['is_active'])
        return Response({
            'success': True,
            'data': None,
            'message': 'Nhân viên đã bị vô hiệu hóa.',
        })

    @action(detail=True, methods=['post'], url_path='register-face')
    def register_face(self, request, pk=None):
        """Save face encoding for an employee."""
        employee = self.get_object()
        serializer = FaceEncodingSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        employee.face_encoding = serializer.validated_data['face_encoding']
        employee.save(update_fields=['face_encoding'])
        return Response({
            'success': True,
            'data': {'face_encoding': employee.face_encoding},
            'message': 'Đã lưu dữ liệu khuôn mặt.',
        })
