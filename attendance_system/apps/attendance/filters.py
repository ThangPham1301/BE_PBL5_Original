import django_filters
from .models import AttendanceLog


class AttendanceLogFilter(django_filters.FilterSet):
    date = django_filters.DateFilter(field_name='date')
    date_from = django_filters.DateFilter(field_name='date', lookup_expr='gte')
    date_to = django_filters.DateFilter(field_name='date', lookup_expr='lte')
    employee_id = django_filters.CharFilter(field_name='employee__employee_id')
    department_id = django_filters.NumberFilter(field_name='employee__department__id')
    status = django_filters.ChoiceFilter(choices=AttendanceLog.Status.choices)

    class Meta:
        model = AttendanceLog
        fields = ['date', 'date_from', 'date_to', 'employee_id', 'department_id', 'status']
