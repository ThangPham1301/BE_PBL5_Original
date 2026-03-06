from django.db import models
from django.conf import settings


class Department(models.Model):
    name = models.CharField(max_length=100, verbose_name='Tên phòng ban')
    manager = models.ForeignKey(
        'Employee',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='managed_departments',
        verbose_name='Trưởng phòng',
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')

    class Meta:
        db_table = 'departments'
        verbose_name = 'Phòng ban'
        verbose_name_plural = 'Phòng ban'

    def __str__(self):
        return self.name


class Employee(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='employee',
        verbose_name='Tài khoản',
    )
    employee_id = models.CharField(
        max_length=20,
        unique=True,
        verbose_name='Mã nhân viên',
    )
    department = models.ForeignKey(
        Department,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='employees',
        verbose_name='Phòng ban',
    )
    position = models.CharField(
        max_length=100,
        blank=True,
        default='',
        verbose_name='Chức vụ',
    )
    phone = models.CharField(
        max_length=20,
        blank=True,
        default='',
        verbose_name='Số điện thoại',
    )
    face_encoding = models.JSONField(
        null=True,
        blank=True,
        verbose_name='Dữ liệu khuôn mặt',
    )
    date_joined = models.DateField(
        verbose_name='Ngày vào làm',
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name='Đang hoạt động',
    )

    class Meta:
        db_table = 'employees'
        verbose_name = 'Nhân viên'
        verbose_name_plural = 'Nhân viên'

    def __str__(self):
        return f"{self.employee_id} - {self.user.get_full_name() or self.user.username}"
