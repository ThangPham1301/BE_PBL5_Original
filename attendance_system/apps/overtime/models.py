from decimal import Decimal

from django.conf import settings
from django.db import models

from apps.employees.models import Employee


class OvertimeRequest(models.Model):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Chờ duyệt'
        APPROVED = 'approved', 'Đã duyệt'
        REJECTED = 'rejected', 'Từ chối'

    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE,
        related_name='overtime_requests',
        verbose_name='Nhân viên',
    )
    date = models.DateField(verbose_name='Ngày tăng ca')
    planned_start_time = models.TimeField(verbose_name='Giờ bắt đầu')
    planned_end_time = models.TimeField(verbose_name='Giờ kết thúc')
    planned_hours = models.DecimalField(
        max_digits=3,
        decimal_places=1,
        editable=False,
        verbose_name='Số giờ dự kiến',
    )
    overtime_rate = models.DecimalField(
        max_digits=2,
        decimal_places=1,
        editable=False,
        verbose_name='Hệ số tăng ca',
    )
    reason = models.TextField(
        blank=True,
        default='',
        verbose_name='Lý do',
    )
    status = models.CharField(
        max_length=10,
        choices=Status.choices,
        default=Status.PENDING,
        verbose_name='Trạng thái',
    )
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_overtime_requests',
        verbose_name='Người duyệt',
    )
    rejection_reason = models.TextField(
        blank=True,
        default='',
        verbose_name='Lý do từ chối',
    )
    reviewed_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name='Thời gian duyệt',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'overtime_requests'
        ordering = ['-date', '-created_at']
        verbose_name = 'Đơn tăng ca'
        verbose_name_plural = 'Đơn tăng ca'

    def __str__(self):
        return f'{self.employee} - {self.date} ({self.get_status_display()})'

    @staticmethod
    def calculate_hours(start_time, end_time):
        start_minutes = start_time.hour * 60 + start_time.minute
        end_minutes = end_time.hour * 60 + end_time.minute
        return Decimal(end_minutes - start_minutes) / Decimal(60)

    @staticmethod
    def calculate_rate(overtime_date):
        return Decimal('2.0') if overtime_date.weekday() >= 5 else Decimal('1.5')

    def save(self, *args, **kwargs):
        self.planned_hours = self.calculate_hours(
            self.planned_start_time,
            self.planned_end_time,
        ).quantize(Decimal('0.1'))
        self.overtime_rate = self.calculate_rate(self.date)
        super().save(*args, **kwargs)
