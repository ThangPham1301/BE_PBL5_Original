from datetime import date

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from apps.employees.models import Employee
from .models import AttendanceLog


class AttendanceHistoryPermissionTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.employee_user = user_model.objects.create_user(
            username='attendance_employee',
            password='test-password',
            role='employee',
        )
        self.employee = Employee.objects.create(
            user=self.employee_user,
            employee_id='ATT-001',
            date_joined=date(2026, 1, 1),
        )
        self.other_employee_user = user_model.objects.create_user(
            username='other_attendance_employee',
            password='test-password',
            role='employee',
        )
        self.other_employee = Employee.objects.create(
            user=self.other_employee_user,
            employee_id='ATT-002',
            date_joined=date(2026, 1, 1),
        )
        self.own_log = AttendanceLog.objects.create(
            employee=self.employee,
            date=date(2026, 6, 5),
            status=AttendanceLog.Status.PRESENT,
        )
        self.other_log = AttendanceLog.objects.create(
            employee=self.other_employee,
            date=date(2026, 6, 5),
            status=AttendanceLog.Status.LATE,
        )

    def test_employee_can_view_only_their_attendance_history(self):
        self.client.force_authenticate(user=self.employee_user)

        list_response = self.client.get('/api/attendance/')
        own_detail_response = self.client.get(f'/api/attendance/{self.own_log.id}/')
        other_detail_response = self.client.get(f'/api/attendance/{self.other_log.id}/')

        self.assertEqual(list_response.status_code, status.HTTP_200_OK)
        self.assertEqual(own_detail_response.status_code, status.HTTP_200_OK)
        self.assertEqual(other_detail_response.status_code, status.HTTP_404_NOT_FOUND)

        listed_ids = {item['id'] for item in list_response.data['results']}
        self.assertEqual(listed_ids, {self.own_log.id})


class SmartOfficeAccessTests(APITestCase):
    def test_all_roles_require_active_check_in(self):
        user_model = get_user_model()

        for index, role in enumerate(('employee', 'manager', 'admin'), start=1):
            with self.subTest(role=role):
                user = user_model.objects.create_user(
                    username=f'smart_office_{role}',
                    password='test-password',
                    role=role,
                )
                employee = Employee.objects.create(
                    user=user,
                    employee_id=f'SMART-{index:03d}',
                    date_joined=date(2026, 1, 1),
                )
                self.client.force_authenticate(user=user)

                before_check_in = self.client.get('/api/attendance/smart-office-access/')
                self.assertEqual(before_check_in.status_code, status.HTTP_200_OK)
                self.assertFalse(before_check_in.data['data']['can_control'])

                log = AttendanceLog.objects.create(
                    employee=employee,
                    date=timezone.localdate(),
                    check_in=timezone.now(),
                    status=AttendanceLog.Status.PRESENT,
                )
                after_check_in = self.client.get('/api/attendance/smart-office-access/')
                self.assertTrue(after_check_in.data['data']['can_control'])

                log.check_out = timezone.now()
                log.save(update_fields=['check_out'])
                after_check_out = self.client.get('/api/attendance/smart-office-access/')
                self.assertFalse(after_check_out.data['data']['can_control'])
                self.assertTrue(after_check_out.data['data']['checked_out'])
