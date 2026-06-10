import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('attendance', '0001_initial'),
        ('shifts', '0003_unique_employee_shift_effective_date'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='attendancelog',
            unique_together=set(),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='employee_shift',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='attendance_logs',
                to='shifts.employeeshift',
                verbose_name='Ca được phân',
            ),
        ),
        migrations.AddConstraint(
            model_name='attendancelog',
            constraint=models.UniqueConstraint(
                fields=('employee', 'date', 'employee_shift'),
                name='unique_employee_attendance_per_shift_date',
            ),
        ),
    ]
