from django.db import models
from apps.employees.models import Employee


class AttendanceLog(models.Model):
    class Status(models.TextChoices):
        PRESENT = 'present', 'Có mặt'
        LATE = 'late', 'Đi trễ'
        ABSENT = 'absent', 'Vắng'
        LEAVE = 'leave', 'Nghỉ phép'

    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE,
        related_name='attendance_logs',
        verbose_name='Nhân viên',
    )
    date = models.DateField(verbose_name='Ngày')
    check_in = models.DateTimeField(null=True, blank=True, verbose_name='Giờ vào')
    check_out = models.DateTimeField(null=True, blank=True, verbose_name='Giờ ra')
    status = models.CharField(
        max_length=10,
        choices=Status.choices,
        default=Status.PRESENT,
        verbose_name='Trạng thái',
    )
    note = models.TextField(blank=True, default='', verbose_name='Ghi chú')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')

    class Meta:
        db_table = 'attendance_logs'
        unique_together = ['employee', 'date']
        verbose_name = 'Bản chấm công'
        verbose_name_plural = 'Bản chấm công'
        ordering = ['-date', '-check_in']

    def __str__(self):
        return f"{self.employee} - {self.date} ({self.get_status_display()})"
