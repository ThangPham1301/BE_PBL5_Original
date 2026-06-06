from datetime import date

from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.test import APITestCase

from apps.employees.models import Employee
from .models import LeaveRequest, LeaveType


class LeaveRequestPermissionTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.admin = user_model.objects.create_user(
            username='leave_admin',
            password='test-password',
            role='admin',
        )
        self.manager = user_model.objects.create_user(
            username='leave_manager',
            password='test-password',
            role='manager',
        )
        self.employee_user = user_model.objects.create_user(
            username='leave_employee',
            password='test-password',
            role='employee',
        )
        self.employee = Employee.objects.create(
            user=self.employee_user,
            employee_id='LEAVE-001',
            date_joined=date(2026, 1, 1),
        )
        self.other_employee_user = user_model.objects.create_user(
            username='other_leave_employee',
            password='test-password',
            role='employee',
        )
        self.other_employee = Employee.objects.create(
            user=self.other_employee_user,
            employee_id='LEAVE-002',
            date_joined=date(2026, 1, 1),
        )
        self.leave_type = LeaveType.objects.create(name='Annual leave', max_days_per_year=12)
        self.leave_request = LeaveRequest.objects.create(
            employee=self.employee,
            leave_type=self.leave_type,
            start_date=date(2026, 6, 10),
            end_date=date(2026, 6, 11),
            reason='Personal',
        )
        self.other_leave_request = LeaveRequest.objects.create(
            employee=self.other_employee,
            leave_type=self.leave_type,
            start_date=date(2026, 7, 10),
            end_date=date(2026, 7, 11),
            reason='Other employee',
        )

    def authenticate(self, user):
        self.client.force_authenticate(user=user)

    def test_employee_can_create_and_view_only_their_leave_requests(self):
        self.authenticate(self.employee_user)

        create_response = self.client.post(
            '/api/leaves/',
            {
                'leave_type': self.leave_type.id,
                'start_date': '2026-06-20',
                'end_date': '2026-06-21',
                'reason': 'Family event',
            },
            format='json',
        )
        list_response = self.client.get('/api/leaves/')
        detail_response = self.client.get(f'/api/leaves/{self.leave_request.id}/')
        other_detail_response = self.client.get(f'/api/leaves/{self.other_leave_request.id}/')

        self.assertEqual(create_response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(list_response.status_code, status.HTTP_200_OK)
        self.assertEqual(detail_response.status_code, status.HTTP_200_OK)
        self.assertEqual(other_detail_response.status_code, status.HTTP_404_NOT_FOUND)
        listed_ids = {item['id'] for item in list_response.data['results']}
        self.assertIn(self.leave_request.id, listed_ids)
        self.assertNotIn(self.other_leave_request.id, listed_ids)

    def test_admin_can_view_and_review_but_cannot_create_leave_request(self):
        self.authenticate(self.admin)

        list_response = self.client.get('/api/leaves/')
        approve_response = self.client.put(f'/api/leaves/{self.leave_request.id}/approve/')
        create_response = self.client.post(
            '/api/leaves/',
            {
                'leave_type': self.leave_type.id,
                'start_date': '2026-06-20',
                'end_date': '2026-06-21',
                'reason': 'Admin request',
            },
            format='json',
        )

        self.assertEqual(list_response.status_code, status.HTTP_200_OK)
        self.assertEqual(approve_response.status_code, status.HTTP_200_OK)
        self.assertEqual(create_response.status_code, status.HTTP_403_FORBIDDEN)

    def test_manager_cannot_create_view_or_review_leave_requests(self):
        self.authenticate(self.manager)

        list_response = self.client.get('/api/leaves/')
        approve_response = self.client.put(f'/api/leaves/{self.leave_request.id}/approve/')
        create_response = self.client.post(
            '/api/leaves/',
            {
                'leave_type': self.leave_type.id,
                'start_date': '2026-06-20',
                'end_date': '2026-06-21',
                'reason': 'Manager request',
            },
            format='json',
        )

        self.assertEqual(list_response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(approve_response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(create_response.status_code, status.HTTP_403_FORBIDDEN)
