from datetime import date, datetime, time
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from apps.employees.models import Employee
from apps.shifts.models import EmployeeShift, Shift
from .views import AttendanceViewSet
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
    def test_employee_and_manager_require_active_check_in(self):
        user_model = get_user_model()

        for index, role in enumerate(('employee', 'manager'), start=1):
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

    def test_admin_can_control_without_employee_or_check_in(self):
        admin = get_user_model().objects.create_user(
            username='smart_office_admin',
            password='test-password',
            role='admin',
        )
        self.client.force_authenticate(user=admin)

        response = self.client.get('/api/attendance/smart-office-access/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['data']['can_control'])
        self.assertTrue(response.data['data']['admin_override'])
        self.assertFalse(response.data['data']['checked_in'])
        self.assertFalse(response.data['data']['checked_out'])


class MultipleShiftAttendanceTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username='multi_attendance_employee',
            password='test-password',
            role='employee',
        )
        self.employee = Employee.objects.create(
            user=self.user,
            employee_id='MULTI-ATT-001',
            date_joined=date(2026, 1, 1),
        )
        morning = Shift.objects.create(
            name='Ca sáng',
            start_time=time(8, 0),
            end_time=time(12, 0),
        )
        afternoon = Shift.objects.create(
            name='Ca chiều',
            start_time=time(12, 0),
            end_time=time(17, 0),
        )
        self.morning_assignment = EmployeeShift.objects.create(
            employee=self.employee,
            shift=morning,
            effective_date=date(2026, 6, 10),
        )
        self.afternoon_assignment = EmployeeShift.objects.create(
            employee=self.employee,
            shift=afternoon,
            effective_date=date(2026, 6, 10),
        )

    def test_employee_can_check_in_to_two_shifts_on_the_same_day(self):
        view = AttendanceViewSet()
        first_check_in = timezone.make_aware(datetime(2026, 6, 10, 7, 55))
        second_check_in = timezone.make_aware(datetime(2026, 6, 10, 12, 0))

        with patch(
            'apps.attendance.views.timezone.localdate',
            return_value=date(2026, 6, 10),
        ), patch(
            'apps.attendance.views.timezone.now',
            return_value=first_check_in,
        ):
            morning_log, morning_error = view._create_check_in_log(
                self.employee
            )
            self.assertIsNone(morning_error)
            morning_log.check_out = timezone.make_aware(
                datetime(2026, 6, 10, 11, 55)
            )
            morning_log.save(update_fields=['check_out'])

        with patch(
            'apps.attendance.views.timezone.localdate',
            return_value=date(2026, 6, 10),
        ), patch(
            'apps.attendance.views.timezone.now',
            return_value=second_check_in,
        ):
            afternoon_log, afternoon_error = view._create_check_in_log(
                self.employee
            )

        self.assertIsNone(afternoon_error)
        self.assertEqual(
            morning_log.employee_shift,
            self.morning_assignment,
        )
        self.assertEqual(
            afternoon_log.employee_shift,
            self.afternoon_assignment,
        )
        self.assertEqual(
            AttendanceLog.objects.filter(
                employee=self.employee,
                date=date(2026, 6, 10),
            ).count(),
            2,
        )

    def test_face_scan_before_shift_end_does_not_check_out(self):
        view = AttendanceViewSet()
        check_in_time = timezone.make_aware(datetime(2026, 6, 10, 8, 0))
        scan_time = timezone.make_aware(datetime(2026, 6, 10, 11, 59))
        log = AttendanceLog.objects.create(
            employee=self.employee,
            employee_shift=self.morning_assignment,
            date=date(2026, 6, 10),
            check_in=check_in_time,
            status=AttendanceLog.Status.PRESENT,
        )

        with patch(
            'apps.attendance.views.timezone.localdate',
            return_value=date(2026, 6, 10),
        ), patch(
            'apps.attendance.views.timezone.now',
            return_value=scan_time,
        ):
            scanned_log, action, error = view._process_attendance_scan(
                self.employee
            )

        log.refresh_from_db()
        self.assertIsNone(scanned_log)
        self.assertIsNone(action)
        self.assertEqual(error.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIsNone(log.check_out)
        self.assertIn('12:00', error.data['message'])

    def test_face_scan_at_shift_end_checks_out_open_shift(self):
        view = AttendanceViewSet()
        check_in_time = timezone.make_aware(datetime(2026, 6, 10, 8, 0))
        scan_time = timezone.make_aware(datetime(2026, 6, 10, 12, 0))
        log = AttendanceLog.objects.create(
            employee=self.employee,
            employee_shift=self.morning_assignment,
            date=date(2026, 6, 10),
            check_in=check_in_time,
            status=AttendanceLog.Status.PRESENT,
        )

        with patch(
            'apps.attendance.views.timezone.localdate',
            return_value=date(2026, 6, 10),
        ), patch(
            'apps.attendance.views.timezone.now',
            return_value=scan_time,
        ):
            scanned_log, action, error = view._process_attendance_scan(
                self.employee
            )

        log.refresh_from_db()
        self.assertIsNone(error)
        self.assertEqual(action, 'check_out')
        self.assertEqual(scanned_log.id, log.id)
        self.assertEqual(log.check_out, scan_time)

    def test_cannot_check_in_after_all_assigned_shifts_have_ended(self):
        view = AttendanceViewSet()
        self.afternoon_assignment.delete()
        scan_time = timezone.make_aware(datetime(2026, 6, 10, 12, 1))

        with patch(
            'apps.attendance.views.timezone.localdate',
            return_value=date(2026, 6, 10),
        ), patch(
            'apps.attendance.views.timezone.now',
            return_value=scan_time,
        ):
            log, error = view._create_check_in_log(self.employee)

        self.assertIsNone(log)
        self.assertEqual(error.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(
            AttendanceLog.objects.filter(employee=self.employee).exists()
        )
