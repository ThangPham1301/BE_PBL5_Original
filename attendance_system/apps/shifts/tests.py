from datetime import date, time

from rest_framework import status
from rest_framework.test import APITestCase

from django.contrib.auth import get_user_model

from apps.employees.models import Department, Employee
from apps.employees.serializers import EmployeeCreateSerializer
from apps.attendance.models import AttendanceLog
from .defaults import DEFAULT_SHIFT_NAME, DEFAULT_SHIFT_WORK_DAYS
from .models import EmployeeShift, Shift
from .serializers import ShiftSerializer


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
        self.assertEqual(len(shift_data), 1)
        self.assertEqual(shift_data[0]['name'], DEFAULT_SHIFT_NAME)
        self.assertEqual(shift_data[0]['start_time'], '08:00:00')
        self.assertEqual(shift_data[0]['end_time'], '17:00:00')
        self.assertEqual(shift_data[0]['check_in_time'], '08:00:00')
        self.assertEqual(shift_data[0]['check_out_time'], '17:00:00')
        self.assertEqual(shift_data[0]['work_days'], DEFAULT_SHIFT_WORK_DAYS)


class ShiftTimeValidationTests(APITestCase):
    def test_shift_accepts_only_hh_mm_ss_time_format(self):
        serializer = ShiftSerializer(data={
            'name': 'Ca hành chính',
            'start_time': '08:00',
            'end_time': '17:00:00',
            'late_threshold': 15,
        })

        self.assertFalse(serializer.is_valid())
        self.assertIn('start_time', serializer.errors)

    def test_shift_end_time_must_be_after_start_time(self):
        serializer = ShiftSerializer(data={
            'name': 'Ca không hợp lệ',
            'start_time': '17:00:00',
            'end_time': '08:00:00',
            'late_threshold': 15,
        })

        self.assertFalse(serializer.is_valid())
        self.assertIn('end_time', serializer.errors)


class MultipleShiftAssignmentTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.admin = user_model.objects.create_user(
            username='shift_admin',
            password='test-password',
            role='admin',
        )
        employee_user = user_model.objects.create_user(
            username='multi_shift_employee',
            password='test-password',
            role='employee',
        )
        self.employee = Employee.objects.create(
            user=employee_user,
            employee_id='MULTI-001',
            date_joined=date(2026, 1, 1),
        )
        self.morning_shift = Shift.objects.create(
            name='Ca sáng',
            start_time=time(8, 0),
            end_time=time(12, 0),
        )
        self.afternoon_shift = Shift.objects.create(
            name='Ca chiều',
            start_time=time(12, 0),
            end_time=time(17, 0),
        )
        self.overlapping_shift = Shift.objects.create(
            name='Ca bị cấn',
            start_time=time(11, 0),
            end_time=time(14, 0),
        )
        self.client.force_authenticate(user=self.admin)

    def assign(self, shift):
        return self.client.post('/api/shifts/assign/', {
            'employee_id': self.employee.id,
            'shift_id': shift.id,
            'effective_date': '2026-06-10',
        })

    def test_touching_shifts_can_be_assigned_on_the_same_schedule(self):
        first_response = self.assign(self.morning_shift)
        second_response = self.assign(self.afternoon_shift)

        self.assertEqual(first_response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(second_response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            EmployeeShift.objects.filter(
                employee=self.employee,
                effective_date=date(2026, 6, 10),
            ).count(),
            2,
        )

    def test_overlapping_shift_is_rejected(self):
        self.assign(self.morning_shift)

        response = self.assign(self.overlapping_shift)

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('cấn giờ', response.data['message'])

    def test_my_shifts_returns_all_shifts_in_latest_schedule(self):
        self.assign(self.morning_shift)
        self.assign(self.afternoon_shift)
        self.client.force_authenticate(user=self.employee.user)

        response = self.client.get('/api/shifts/my/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            [item['name'] for item in response.data['data']],
            ['Ca sáng', 'Ca chiều'],
        )

    def test_new_schedule_date_keeps_previous_non_overlapping_shifts(self):
        self.assign(self.morning_shift)

        response = self.client.post('/api/shifts/assign/', {
            'employee_id': self.employee.id,
            'shift_id': self.afternoon_shift.id,
            'effective_date': '2026-06-11',
        })

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        inherited_schedule = EmployeeShift.objects.filter(
            employee=self.employee,
            effective_date=date(2026, 6, 11),
        )
        self.assertEqual(
            set(inherited_schedule.values_list('shift__name', flat=True)),
            {'Ca sáng', 'Ca chiều'},
        )

    def test_rejected_new_schedule_does_not_copy_previous_assignments(self):
        self.assign(self.morning_shift)

        response = self.client.post('/api/shifts/assign/', {
            'employee_id': self.employee.id,
            'shift_id': self.overlapping_shift.id,
            'effective_date': '2026-06-11',
        })

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(
            EmployeeShift.objects.filter(
                employee=self.employee,
                effective_date=date(2026, 6, 11),
            ).exists()
        )

    def test_admin_can_list_and_remove_employee_shift(self):
        assignment = EmployeeShift.objects.create(
            employee=self.employee,
            shift=self.morning_shift,
            effective_date=date(2026, 6, 10),
        )

        list_response = self.client.get(
            f'/api/shifts/assignments/?employee_id={self.employee.id}'
        )
        delete_response = self.client.delete(
            f'/api/shifts/assignments/{assignment.id}/'
        )

        self.assertEqual(list_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(list_response.data['data']), 1)
        self.assertEqual(
            list_response.data['data'][0]['employee_code'],
            self.employee.employee_id,
        )
        self.assertEqual(delete_response.status_code, status.HTTP_200_OK)
        self.assertFalse(
            EmployeeShift.objects.filter(pk=assignment.id).exists()
        )

    def test_assignment_with_attendance_cannot_be_removed(self):
        assignment = EmployeeShift.objects.create(
            employee=self.employee,
            shift=self.morning_shift,
            effective_date=date(2026, 6, 10),
        )
        AttendanceLog.objects.create(
            employee=self.employee,
            employee_shift=assignment,
            date=date(2026, 6, 10),
            status=AttendanceLog.Status.PRESENT,
        )

        response = self.client.delete(
            f'/api/shifts/assignments/{assignment.id}/'
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertTrue(
            EmployeeShift.objects.filter(pk=assignment.id).exists()
        )

    def test_employee_cannot_remove_an_assignment(self):
        assignment = EmployeeShift.objects.create(
            employee=self.employee,
            shift=self.morning_shift,
            effective_date=date(2026, 6, 10),
        )
        self.client.force_authenticate(user=self.employee.user)

        response = self.client.delete(
            f'/api/shifts/assignments/{assignment.id}/'
        )

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertTrue(
            EmployeeShift.objects.filter(pk=assignment.id).exists()
        )

    def test_manager_cannot_remove_assignment_from_another_department(self):
        manager_department = Department.objects.create(name='Phòng quản lý')
        other_department = Department.objects.create(name='Phòng khác')
        manager_user = get_user_model().objects.create_user(
            username='assignment_manager',
            password='test-password',
            role='manager',
        )
        Employee.objects.create(
            user=manager_user,
            employee_id='MGR-ASSIGN',
            department=manager_department,
            date_joined=date(2026, 1, 1),
        )
        self.employee.department = other_department
        self.employee.save(update_fields=['department'])
        assignment = EmployeeShift.objects.create(
            employee=self.employee,
            shift=self.morning_shift,
            effective_date=date(2026, 6, 10),
        )
        self.client.force_authenticate(user=manager_user)

        response = self.client.delete(
            f'/api/shifts/assignments/{assignment.id}/'
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertTrue(
            EmployeeShift.objects.filter(pk=assignment.id).exists()
        )
