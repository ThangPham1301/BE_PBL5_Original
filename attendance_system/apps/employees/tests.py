from rest_framework.test import APITestCase

from .serializers import EmployeeCreateSerializer, EmployeeUpdateSerializer


class AutomaticEmployeeIdTests(APITestCase):
    def create_employee(self, username, role='employee', employee_id=None):
        data = {
            'user': {
                'username': username,
                'password': 'test-password',
                'role': role,
                'first_name': 'Test',
                'last_name': 'User',
            },
        }
        if employee_id:
            data['employee_id'] = employee_id

        serializer = EmployeeCreateSerializer(data=data)
        serializer.is_valid(raise_exception=True)
        return serializer.save()

    def test_employee_id_is_generated_and_manual_value_is_ignored(self):
        employee = self.create_employee(
            'automatic_employee',
            employee_id='MANUAL-001',
        )

        self.assertEqual(employee.employee_id, f'NV{employee.pk:05d}')

    def test_manager_and_employee_receive_unique_role_prefixes(self):
        employee = self.create_employee('automatic_employee_2')
        manager = self.create_employee('automatic_manager', role='manager')

        self.assertEqual(employee.employee_id, f'NV{employee.pk:05d}')
        self.assertEqual(manager.employee_id, f'MGR{manager.pk:05d}')
        self.assertNotEqual(employee.employee_id, manager.employee_id)

    def test_employee_id_cannot_be_changed_by_update_serializer(self):
        employee = self.create_employee('immutable_employee_id')
        original_employee_id = employee.employee_id
        serializer = EmployeeUpdateSerializer(
            employee,
            data={'employee_id': 'CHANGED-001', 'phone': '0901000000'},
            partial=True,
        )
        serializer.is_valid(raise_exception=True)

        updated_employee = serializer.save()

        self.assertEqual(updated_employee.employee_id, original_employee_id)
        self.assertEqual(updated_employee.phone, '0901000000')
