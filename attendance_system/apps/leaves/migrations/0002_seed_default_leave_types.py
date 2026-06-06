from django.db import migrations


DEFAULT_LEAVE_TYPES = (
    ('Nghỉ phép năm', 12),
    ('Nghỉ ốm', 30),
    ('Nghỉ không lương', 365),
    ('Nghỉ thai sản', 180),
    ('Nghỉ kết hôn', 3),
    ('Nghỉ tang', 3),
    ('Nghỉ việc riêng', 5),
)


def seed_default_leave_types(apps, schema_editor):
    leave_type_model = apps.get_model('leaves', 'LeaveType')
    for name, max_days_per_year in DEFAULT_LEAVE_TYPES:
        leave_type_model.objects.update_or_create(
            name=name,
            defaults={'max_days_per_year': max_days_per_year},
        )


def remove_default_leave_types(apps, schema_editor):
    leave_type_model = apps.get_model('leaves', 'LeaveType')
    leave_type_model.objects.filter(
        name__in=[name for name, _ in DEFAULT_LEAVE_TYPES],
        leave_requests__isnull=True,
    ).delete()


class Migration(migrations.Migration):
    dependencies = [
        ('leaves', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(
            seed_default_leave_types,
            reverse_code=remove_default_leave_types,
        ),
    ]
