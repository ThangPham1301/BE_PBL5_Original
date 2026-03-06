from datetime import date, time

from django.core.management.base import BaseCommand

from apps.accounts.models import CustomUser
from apps.employees.models import Department, Employee
from apps.shifts.models import Shift, EmployeeShift
from apps.leaves.models import LeaveType


class Command(BaseCommand):
    help = 'Seed database with sample data: departments, users, employees, shifts, leave types'

    def handle(self, *args, **options):
        self.stdout.write('Seeding database...')

        # ─── Leave Types ─────────────────────────────────
        lt_annual, _ = LeaveType.objects.get_or_create(
            name='Nghỉ phép năm', defaults={'max_days_per_year': 12}
        )
        lt_sick, _ = LeaveType.objects.get_or_create(
            name='Nghỉ ốm', defaults={'max_days_per_year': 30}
        )
        self.stdout.write(self.style.SUCCESS('  ✓ Leave types created'))

        # ─── Shifts ──────────────────────────────────────
        shift_morning, _ = Shift.objects.get_or_create(
            name='Ca sáng',
            defaults={
                'start_time': time(8, 0),
                'end_time': time(12, 0),
                'late_threshold': 15,
            },
        )
        shift_afternoon, _ = Shift.objects.get_or_create(
            name='Ca chiều',
            defaults={
                'start_time': time(13, 0),
                'end_time': time(17, 0),
                'late_threshold': 15,
            },
        )
        self.stdout.write(self.style.SUCCESS('  ✓ Shifts created'))

        # ─── Admin User ─────────────────────────────────
        admin_user, created = CustomUser.objects.get_or_create(
            username='admin',
            defaults={
                'email': 'admin@company.com',
                'first_name': 'System',
                'last_name': 'Admin',
                'role': CustomUser.Role.ADMIN,
                'is_staff': True,
                'is_superuser': True,
            },
        )
        if created:
            admin_user.set_password('admin123')
            admin_user.save()
        self.stdout.write(self.style.SUCCESS('  ✓ Admin user created (admin / admin123)'))

        # ─── Departments (created without manager first) ─
        dept_it, _ = Department.objects.get_or_create(name='Phòng Công nghệ thông tin')
        dept_hr, _ = Department.objects.get_or_create(name='Phòng Nhân sự')
        self.stdout.write(self.style.SUCCESS('  ✓ Departments created'))

        # ─── Manager Users + Employees ───────────────────
        mgr1_user, created = CustomUser.objects.get_or_create(
            username='manager_it',
            defaults={
                'email': 'manager_it@company.com',
                'first_name': 'Nguyễn Văn',
                'last_name': 'An',
                'role': CustomUser.Role.MANAGER,
            },
        )
        if created:
            mgr1_user.set_password('manager123')
            mgr1_user.save()

        mgr1_emp, _ = Employee.objects.get_or_create(
            user=mgr1_user,
            defaults={
                'employee_id': 'MGR001',
                'department': dept_it,
                'position': 'Trưởng phòng IT',
                'phone': '0901000001',
                'date_joined': date(2023, 1, 15),
            },
        )
        dept_it.manager = mgr1_emp
        dept_it.save()

        mgr2_user, created = CustomUser.objects.get_or_create(
            username='manager_hr',
            defaults={
                'email': 'manager_hr@company.com',
                'first_name': 'Trần Thị',
                'last_name': 'Bình',
                'role': CustomUser.Role.MANAGER,
            },
        )
        if created:
            mgr2_user.set_password('manager123')
            mgr2_user.save()

        mgr2_emp, _ = Employee.objects.get_or_create(
            user=mgr2_user,
            defaults={
                'employee_id': 'MGR002',
                'department': dept_hr,
                'position': 'Trưởng phòng Nhân sự',
                'phone': '0901000002',
                'date_joined': date(2023, 2, 1),
            },
        )
        dept_hr.manager = mgr2_emp
        dept_hr.save()
        self.stdout.write(self.style.SUCCESS('  ✓ Managers created'))

        # ─── Employee Users ──────────────────────────────
        employees_data = [
            {
                'username': 'nv001', 'email': 'nv001@company.com',
                'first_name': 'Lê Minh', 'last_name': 'Châu',
                'employee_id': 'NV001', 'department': dept_it,
                'position': 'Developer', 'phone': '0901000011',
                'date_joined': date(2023, 6, 1),
            },
            {
                'username': 'nv002', 'email': 'nv002@company.com',
                'first_name': 'Phạm Thanh', 'last_name': 'Dung',
                'employee_id': 'NV002', 'department': dept_it,
                'position': 'Developer', 'phone': '0901000012',
                'date_joined': date(2023, 7, 15),
            },
            {
                'username': 'nv003', 'email': 'nv003@company.com',
                'first_name': 'Hoàng Văn', 'last_name': 'Em',
                'employee_id': 'NV003', 'department': dept_it,
                'position': 'Tester', 'phone': '0901000013',
                'date_joined': date(2024, 1, 10),
            },
            {
                'username': 'nv004', 'email': 'nv004@company.com',
                'first_name': 'Đỗ Thị', 'last_name': 'Phượng',
                'employee_id': 'NV004', 'department': dept_hr,
                'position': 'HR Specialist', 'phone': '0901000014',
                'date_joined': date(2023, 8, 1),
            },
            {
                'username': 'nv005', 'email': 'nv005@company.com',
                'first_name': 'Vũ Quang', 'last_name': 'Giang',
                'employee_id': 'NV005', 'department': dept_hr,
                'position': 'Recruiter', 'phone': '0901000015',
                'date_joined': date(2024, 3, 1),
            },
        ]

        created_employees = []
        for emp_data in employees_data:
            user, created = CustomUser.objects.get_or_create(
                username=emp_data['username'],
                defaults={
                    'email': emp_data['email'],
                    'first_name': emp_data['first_name'],
                    'last_name': emp_data['last_name'],
                    'role': CustomUser.Role.EMPLOYEE,
                },
            )
            if created:
                user.set_password('employee123')
                user.save()

            emp, _ = Employee.objects.get_or_create(
                user=user,
                defaults={
                    'employee_id': emp_data['employee_id'],
                    'department': emp_data['department'],
                    'position': emp_data['position'],
                    'phone': emp_data['phone'],
                    'date_joined': emp_data['date_joined'],
                },
            )
            created_employees.append(emp)

        self.stdout.write(self.style.SUCCESS('  ✓ 5 Employees created'))

        # ─── Assign shifts ───────────────────────────────
        all_employees = [mgr1_emp, mgr2_emp] + created_employees
        for emp in all_employees:
            EmployeeShift.objects.get_or_create(
                employee=emp,
                shift=shift_morning,
                defaults={'effective_date': date(2025, 1, 1)},
            )
        self.stdout.write(self.style.SUCCESS('  ✓ Shifts assigned to all employees'))

        self.stdout.write(self.style.SUCCESS('\nSeed completed successfully!'))
        self.stdout.write('Accounts:')
        self.stdout.write('  admin       / admin123     (role: admin)')
        self.stdout.write('  manager_it  / manager123   (role: manager, IT)')
        self.stdout.write('  manager_hr  / manager123   (role: manager, HR)')
        self.stdout.write('  nv001-nv005 / employee123  (role: employee)')
