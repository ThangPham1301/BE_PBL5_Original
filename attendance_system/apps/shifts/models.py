from django.db import models
from apps.employees.models import Employee


class Shift(models.Model):
    name = models.CharField(max_length=100, verbose_name='Tên ca')
    start_time = models.TimeField(verbose_name='Giờ bắt đầu')
    end_time = models.TimeField(verbose_name='Giờ kết thúc')
    late_threshold = models.PositiveIntegerField(
        default=15,
        verbose_name='Ngưỡng đi trễ (phút)',
        help_text='Số phút cho phép đến trễ trước khi bị tính trễ.',
    )

    class Meta:
        db_table = 'shifts'
        verbose_name = 'Ca làm việc'
        verbose_name_plural = 'Ca làm việc'

    def __str__(self):
        return f"{self.name} ({self.start_time} - {self.end_time})"


class EmployeeShift(models.Model):
    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE,
        related_name='employee_shifts',
        verbose_name='Nhân viên',
    )
    shift = models.ForeignKey(
        Shift,
        on_delete=models.CASCADE,
        related_name='employee_shifts',
        verbose_name='Ca làm việc',
    )
    effective_date = models.DateField(verbose_name='Ngày hiệu lực')

    class Meta:
        db_table = 'employee_shifts'
        verbose_name = 'Phân ca'
        verbose_name_plural = 'Phân ca'
        ordering = ['-effective_date']

    def __str__(self):
        return f"{self.employee} - {self.shift} (from {self.effective_date})"
