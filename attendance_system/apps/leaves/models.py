from django.db import models
from apps.employees.models import Employee


class LeaveType(models.Model):
    name = models.CharField(max_length=100, verbose_name='Loại nghỉ phép')
    max_days_per_year = models.PositiveIntegerField(
        default=12,
        verbose_name='Số ngày tối đa/năm',
    )

    class Meta:
        db_table = 'leave_types'
        verbose_name = 'Loại nghỉ phép'
        verbose_name_plural = 'Loại nghỉ phép'

    def __str__(self):
        return self.name


class LeaveRequest(models.Model):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Chờ duyệt'
        APPROVED = 'approved', 'Đã duyệt'
        REJECTED = 'rejected', 'Từ chối'

    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE,
        related_name='leave_requests',
        verbose_name='Nhân viên',
    )
    leave_type = models.ForeignKey(
        LeaveType,
        on_delete=models.CASCADE,
        related_name='leave_requests',
        verbose_name='Loại nghỉ phép',
    )
    start_date = models.DateField(verbose_name='Ngày bắt đầu')
    end_date = models.DateField(verbose_name='Ngày kết thúc')
    reason = models.TextField(blank=True, default='', verbose_name='Lý do')
    status = models.CharField(
        max_length=10,
        choices=Status.choices,
        default=Status.PENDING,
        verbose_name='Trạng thái',
    )
    approved_by = models.ForeignKey(
        Employee,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='approved_leaves',
        verbose_name='Người duyệt',
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')

    class Meta:
        db_table = 'leave_requests'
        verbose_name = 'Đơn nghỉ phép'
        verbose_name_plural = 'Đơn nghỉ phép'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.employee} - {self.leave_type} ({self.start_date} → {self.end_date})"

    @property
    def total_days(self):
        return (self.end_date - self.start_date).days + 1
