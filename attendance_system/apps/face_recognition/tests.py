import tempfile
from datetime import date, datetime, time
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from apps.attendance.models import AttendanceLog
from apps.employees.models import Employee
from apps.shifts.models import EmployeeShift, Shift


class FaceAttendanceIntegrationTests(APITestCase):
    def setUp(self):
        self.media_directory = tempfile.TemporaryDirectory()
        self.settings_override = override_settings(
            MEDIA_ROOT=self.media_directory.name,
        )
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)
        self.addCleanup(self.media_directory.cleanup)

        user = get_user_model().objects.create_user(
            username='face_attendance_employee',
            password='test-password',
            role='employee',
        )
        self.employee = Employee.objects.create(
            user=user,
            employee_id='FACE-ATT-001',
            date_joined=date(2026, 1, 1),
        )
        shift = Shift.objects.create(
            name='Ca sáng',
            start_time=time(8, 0),
            end_time=time(12, 0),
        )
        self.assignment = EmployeeShift.objects.create(
            employee=self.employee,
            shift=shift,
            effective_date=date(2026, 6, 10),
        )

    def test_recognize_face_checks_out_at_shift_end(self):
        check_in_time = timezone.make_aware(datetime(2026, 6, 10, 8, 0))
        scan_time = timezone.make_aware(datetime(2026, 6, 10, 12, 0))
        attendance_log = AttendanceLog.objects.create(
            employee=self.employee,
            employee_shift=self.assignment,
            date=date(2026, 6, 10),
            check_in=check_in_time,
            status=AttendanceLog.Status.PRESENT,
        )

        with patch(
            'apps.face_recognition.views.FaceRecognitionService.process_image',
            return_value=(True, self.employee.id, 0.98),
        ), patch(
            'apps.attendance.views.timezone.localdate',
            return_value=date(2026, 6, 10),
        ), patch(
            'apps.attendance.views.timezone.now',
            return_value=scan_time,
        ):
            response = self.client.post(
                '/api/face/recognize/',
                {
                    'file': SimpleUploadedFile(
                        'face.jpg',
                        b'fake-image-content',
                        content_type='image/jpeg',
                    ),
                },
                format='multipart',
            )

        attendance_log.refresh_from_db()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['attendance_action'], 'check_out')
        self.assertEqual(attendance_log.check_out, scan_time)
