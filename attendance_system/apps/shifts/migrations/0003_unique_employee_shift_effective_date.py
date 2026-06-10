from django.db import migrations, models


def remove_duplicate_assignments(apps, schema_editor):
    EmployeeShift = apps.get_model('shifts', 'EmployeeShift')
    duplicates = (
        EmployeeShift.objects
        .values('employee_id', 'shift_id', 'effective_date')
        .annotate(total=models.Count('id'))
        .filter(total__gt=1)
    )
    for duplicate in duplicates:
        ids = list(
            EmployeeShift.objects
            .filter(
                employee_id=duplicate['employee_id'],
                shift_id=duplicate['shift_id'],
                effective_date=duplicate['effective_date'],
            )
            .order_by('id')
            .values_list('id', flat=True)
        )
        EmployeeShift.objects.filter(id__in=ids[1:]).delete()


class Migration(migrations.Migration):
    dependencies = [
        ('shifts', '0002_shift_work_days_and_default_shift'),
    ]

    operations = [
        migrations.RunPython(
            remove_duplicate_assignments,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.AddConstraint(
            model_name='employeeshift',
            constraint=models.UniqueConstraint(
                fields=('employee', 'shift', 'effective_date'),
                name='unique_employee_shift_effective_date',
            ),
        ),
        migrations.AlterModelOptions(
            name='employeeshift',
            options={
                'ordering': ['-effective_date', 'shift__start_time'],
                'verbose_name': 'Phân ca',
                'verbose_name_plural': 'Phân ca',
            },
        ),
    ]
