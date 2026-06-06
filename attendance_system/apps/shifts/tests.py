from datetime import date, time

from rest_framework import status
from rest_framework.test import APITestCase

from apps.employees.serializers import EmployeeCreateSerializer
from .defaults import DEFAULT_SHIFT_NAME, DEFAULT_SHIFT_WORK_DAYS
from .models import EmployeeShift


class DefaultEmployeeShiftTests(APITestCase):
    def test_new_employee_receives_default_weekday_shift(self):
        serializer = EmployeeCreateSerializer(data={
            'user': {
                'username': 'new_shift_employee',
                'email': 'shift@example.com',
                'first_name': 'New',
                'last_name': 'Employee',
                'role': 'employee',
                'password': 'test-password',
            },
            'employee_id': 'SHIFT-001',
            'date_joined': '2026-06-06',
        })
        serializer.is_valid(raise_exception=True)
        employee = serializer.save()

        assignment = EmployeeShift.objects.select_related('shift').get(employee=employee)

        self.assertEqual(assignment.shift.name, DEFAULT_SHIFT_NAME)
        self.assertEqual(assignment.shift.start_time, time(8, 0))
        self.assertEqual(assignment.shift.end_time, time(17, 0))
        self.assertEqual(assignment.shift.work_days, DEFAULT_SHIFT_WORK_DAYS)

    def test_employee_can_get_their_current_shift(self):
        serializer = EmployeeCreateSerializer(data={
            'user': {
                'username': 'api_shift_employee',
                'email': 'api-shift@example.com',
                'first_name': 'API',
                'last_name': 'Employee',
                'role': 'employee',
                'password': 'test-password',
            },
            'employee_id': 'SHIFT-002',
            'date_joined': str(date.today()),
        })
        serializer.is_valid(raise_exception=True)
        employee = serializer.save()

        self.client.force_authenticate(user=employee.user)
        response = self.client.get('/api/shifts/my/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        shift_data = response.data['data']
        self.assertEqual(shift_data['name'], DEFAULT_SHIFT_NAME)
        self.assertEqual(shift_data['start_time'], '08:00:00')
        self.assertEqual(shift_data['end_time'], '17:00:00')
        self.assertEqual(shift_data['work_days'], DEFAULT_SHIFT_WORK_DAYS)
