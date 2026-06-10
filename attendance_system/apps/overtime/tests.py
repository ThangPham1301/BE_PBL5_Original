from datetime import date, time
from unittest.mock import patch

from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.test import APITestCase

from apps.employees.models import Department, Employee

from .models import OvertimeRequest


class OvertimeRequestTests(APITestCase):
    def setUp(self):
        self.localdate_patcher = patch(
            'apps.overtime.serializers.timezone.localdate',
            return_value=date(2026, 6, 10),
        )
        self.localdate_patcher.start()
        self.addCleanup(self.localdate_patcher.stop)

        user_model = get_user_model()
        self.department = Department.objects.create(name='Engineering')
        self.other_department = Department.objects.create(name='Sales')

        self.admin = user_model.objects.create_user(
            username='overtime_admin',
            password='test-password',
            role='admin',
        )
        self.manager_user = user_model.objects.create_user(
            username='overtime_manager',
            password='test-password',
            role='manager',
        )
        self.manager = Employee.objects.create(
            user=self.manager_user,
            employee_id='OT-MANAGER',
            department=self.department,
            date_joined=date(2026, 1, 1),
        )
        self.employee_user = user_model.objects.create_user(
            username='overtime_employee',
            password='test-password',
            role='employee',
        )
        self.employee = Employee.objects.create(
            user=self.employee_user,
            employee_id='OT-EMPLOYEE',
            department=self.department,
            date_joined=date(2026, 1, 1),
        )
        self.other_employee_user = user_model.objects.create_user(
            username='overtime_other_employee',
            password='test-password',
            role='employee',
        )
        self.other_employee = Employee.objects.create(
            user=self.other_employee_user,
            employee_id='OT-OTHER',
            department=self.other_department,
            date_joined=date(2026, 1, 1),
        )

    def authenticate(self, user):
        self.client.force_authenticate(user=user)

    def create_request(
        self,
        employee=None,
        overtime_date=date(2026, 6, 13),
        start_time=time(17, 0),
        end_time=time(19, 0),
    ):
        return OvertimeRequest.objects.create(
            employee=employee or self.employee,
            date=overtime_date,
            planned_start_time=start_time,
            planned_end_time=end_time,
            reason='Release support',
        )

    def test_employee_creates_request_and_backend_calculates_hours_and_rate(self):
        self.authenticate(self.employee_user)

        response = self.client.post(
            '/api/overtime-requests/',
            {
                'date': '2026-06-13',
                'planned_start_time': '17:00',
                'planned_end_time': '19:00',
                'planned_hours': 99,
                'reason': 'Release support',
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        overtime_request = OvertimeRequest.objects.get()
        self.assertEqual(overtime_request.employee, self.employee)
        self.assertEqual(str(overtime_request.planned_hours), '2.0')
        self.assertEqual(str(overtime_request.overtime_rate), '2.0')
        self.assertEqual(response.data['planned_hours'], '2.0')

    def test_request_must_be_at_most_two_hours(self):
        self.authenticate(self.employee_user)

        response = self.client.post(
            '/api/overtime-requests/',
            {
                'date': '2026-06-12',
                'planned_start_time': '17:00',
                'planned_end_time': '19:01',
                'reason': 'Too long',
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('2 tiếng', response.data['non_field_errors'][0])

    def test_overtime_date_cannot_be_in_the_past(self):
        self.authenticate(self.employee_user)

        response = self.client.post(
            '/api/overtime-requests/',
            {
                'date': '2026-06-09',
                'planned_start_time': '17:00',
                'planned_end_time': '18:00',
                'reason': '',
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('ngày hiện tại', response.data['date'][0])

    def test_reason_can_be_blank(self):
        self.authenticate(self.employee_user)

        response = self.client.post(
            '/api/overtime-requests/',
            {
                'date': '2026-06-10',
                'planned_start_time': '17:00',
                'planned_end_time': '18:00',
                'reason': '',
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(OvertimeRequest.objects.get().reason, '')

    def test_employee_cannot_create_overlapping_request(self):
        self.create_request(
            overtime_date=date(2026, 6, 12),
            start_time=time(17, 0),
            end_time=time(18, 0),
        )
        self.authenticate(self.employee_user)

        response = self.client.post(
            '/api/overtime-requests/',
            {
                'date': '2026-06-12',
                'planned_start_time': '17:30',
                'planned_end_time': '18:30',
                'reason': 'Overlapping work',
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('bị trùng', response.data['non_field_errors'][0])

    def test_total_overtime_must_be_at_most_two_hours_per_day(self):
        self.create_request(
            overtime_date=date(2026, 6, 12),
            start_time=time(17, 0),
            end_time=time(18, 0),
        )
        self.authenticate(self.employee_user)

        response = self.client.post(
            '/api/overtime-requests/',
            {
                'date': '2026-06-12',
                'planned_start_time': '18:00',
                'planned_end_time': '19:30',
                'reason': 'Additional work',
            },
            format='json',
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            'Tổng thời gian',
            response.data['non_field_errors'][0],
        )

    def test_employee_only_sees_own_requests(self):
        own_request = self.create_request()
        self.create_request(
            employee=self.other_employee,
            overtime_date=date(2026, 6, 14),
        )
        self.authenticate(self.employee_user)

        response = self.client.get('/api/overtime-requests/')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            {item['id'] for item in response.data['results']},
            {own_request.id},
        )

    def test_request_cannot_be_updated_directly(self):
        overtime_request = self.create_request()
        self.authenticate(self.employee_user)

        response = self.client.put(
            f'/api/overtime-requests/{overtime_request.id}/',
            {
                'date': '2026-06-15',
                'planned_start_time': '17:00',
                'planned_end_time': '18:00',
                'reason': 'Changed reason',
            },
            format='json',
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_405_METHOD_NOT_ALLOWED,
        )

    def test_manager_only_reviews_requests_in_their_department(self):
        own_department_request = self.create_request()
        other_department_request = self.create_request(
            employee=self.other_employee,
            overtime_date=date(2026, 6, 14),
        )
        self.authenticate(self.manager_user)

        list_response = self.client.get('/api/overtime-requests/')
        approve_response = self.client.put(
            f'/api/overtime-requests/{own_department_request.id}/approve/',
        )
        forbidden_response = self.client.put(
            f'/api/overtime-requests/{other_department_request.id}/approve/',
        )

        self.assertEqual(list_response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            {item['id'] for item in list_response.data['results']},
            {own_department_request.id},
        )
        self.assertEqual(approve_response.status_code, status.HTTP_200_OK)
        self.assertEqual(forbidden_response.status_code, status.HTTP_404_NOT_FOUND)
        own_department_request.refresh_from_db()
        self.assertEqual(
            own_department_request.status,
            OvertimeRequest.Status.APPROVED,
        )
        self.assertEqual(own_department_request.reviewed_by, self.manager_user)

    def test_admin_can_reject_but_cannot_review_request_twice(self):
        overtime_request = self.create_request()
        self.authenticate(self.admin)

        reject_response = self.client.put(
            f'/api/overtime-requests/{overtime_request.id}/reject/',
            {'reason': 'Không đủ nhân sự giám sát'},
            format='json',
        )
        second_review_response = self.client.put(
            f'/api/overtime-requests/{overtime_request.id}/approve/',
        )

        self.assertEqual(reject_response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            reject_response.data['data']['rejection_reason'],
            'Không đủ nhân sự giám sát',
        )
        self.assertEqual(
            second_review_response.status_code,
            status.HTTP_400_BAD_REQUEST,
        )

    def test_admin_and_manager_cannot_create_requests(self):
        payload = {
            'date': '2026-06-12',
            'planned_start_time': '17:00',
            'planned_end_time': '18:00',
            'reason': 'Invalid role',
        }
        for user in (self.admin, self.manager_user):
            with self.subTest(role=user.role):
                self.authenticate(user)
                response = self.client.post(
                    '/api/overtime-requests/',
                    payload,
                    format='json',
                )
                self.assertEqual(
                    response.status_code,
                    status.HTTP_403_FORBIDDEN,
                )
