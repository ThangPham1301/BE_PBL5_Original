from datetime import date, time

import apps.shifts.models
from django.db import migrations, models


DEFAULT_SHIFT_NAME = 'Ca hành chính'
DEFAULT_WORK_DAYS = [0, 1, 2, 3, 4]
DEFAULT_EFFECTIVE_DATE = date(2026, 6, 6)


def create_default_shift_and_assign_employees(apps, schema_editor):
    shift_model = apps.get_model('shifts', 'Shift')
    employee_shift_model = apps.get_model('shifts', 'EmployeeShift')
    employee_model = apps.get_model('employees', 'Employee')

    shift, _ = shift_model.objects.update_or_create(
        name=DEFAULT_SHIFT_NAME,
        defaults={
            'start_time': time(8, 0),
            'end_time': time(17, 0),
            'work_days': DEFAULT_WORK_DAYS,
            'late_threshold': 15,
        },
    )

    employees = employee_model.objects.filter(user__role='employee', is_active=True)
    for employee in employees:
        employee_shift_model.objects.get_or_create(
            employee=employee,
            shift=shift,
            effective_date=DEFAULT_EFFECTIVE_DATE,
        )


def remove_default_assignments(apps, schema_editor):
    shift_model = apps.get_model('shifts', 'Shift')
    employee_shift_model = apps.get_model('shifts', 'EmployeeShift')

    shifts = shift_model.objects.filter(name=DEFAULT_SHIFT_NAME)
    employee_shift_model.objects.filter(
        shift__in=shifts,
        effective_date=DEFAULT_EFFECTIVE_DATE,
    ).delete()
    shifts.filter(employee_shifts__isnull=True).delete()


class Migration(migrations.Migration):
    dependencies = [
        ('shifts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='shift',
            name='work_days',
            field=models.JSONField(
                default=apps.shifts.models.default_work_days,
                verbose_name='Ngày làm việc',
            ),
        ),
        migrations.RunPython(
            create_default_shift_and_assign_employees,
            reverse_code=remove_default_assignments,
        ),
    ]
