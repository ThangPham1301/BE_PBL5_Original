from django.contrib import admin
from .models import Shift, EmployeeShift


@admin.register(Shift)
class ShiftAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'start_time', 'end_time', 'work_days', 'late_threshold')


@admin.register(EmployeeShift)
class EmployeeShiftAdmin(admin.ModelAdmin):
    list_display = ('id', 'employee', 'shift', 'effective_date')
    list_filter = ('shift', 'effective_date')
