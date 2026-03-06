from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.permissions import IsAdmin, IsAdminOrManager, IsAuthenticated
from apps.employees.models import Employee
from .models import Shift, EmployeeShift
from .serializers import ShiftSerializer, EmployeeShiftSerializer, AssignShiftSerializer


class ShiftViewSet(viewsets.ModelViewSet):
    queryset = Shift.objects.all().order_by('start_time')
    serializer_class = ShiftSerializer

    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [IsAdmin()]
        return [IsAuthenticated()]

    @action(detail=False, methods=['post'], url_path='assign')
    def assign(self, request):
        """Assign a shift to an employee."""
        serializer = AssignShiftSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            employee = Employee.objects.get(pk=data['employee_id'])
        except Employee.DoesNotExist:
            return Response({
                'success': False,
                'data': None,
                'message': 'Nhân viên không tồn tại.',
            }, status=status.HTTP_404_NOT_FOUND)

        try:
            shift = Shift.objects.get(pk=data['shift_id'])
        except Shift.DoesNotExist:
            return Response({
                'success': False,
                'data': None,
                'message': 'Ca làm việc không tồn tại.',
            }, status=status.HTTP_404_NOT_FOUND)

        emp_shift = EmployeeShift.objects.create(
            employee=employee,
            shift=shift,
            effective_date=data['effective_date'],
        )
        return Response({
            'success': True,
            'data': EmployeeShiftSerializer(emp_shift).data,
            'message': 'Đã gán ca làm việc cho nhân viên.',
        }, status=status.HTTP_201_CREATED)
