from django.contrib import admin

from .models import OvertimeRequest


@admin.register(OvertimeRequest)
class OvertimeRequestAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'employee',
        'date',
        'planned_start_time',
        'planned_end_time',
        'planned_hours',
        'overtime_rate',
        'status',
        'reviewed_by',
    )
    list_filter = ('status', 'date', 'overtime_rate')
    search_fields = (
        'employee__employee_id',
        'employee__user__username',
        'reason',
    )
