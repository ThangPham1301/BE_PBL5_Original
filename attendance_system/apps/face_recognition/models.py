from django.db import models

from apps.employees.models import Employee


class FaceEmbedding(models.Model):
    POSE_FRONT = 'front'
    POSE_LEFT = 'left'
    POSE_RIGHT = 'right'
    POSE_UP = 'up'
    POSE_DOWN = 'down'
    POSE_CHOICES = [
        (POSE_FRONT, 'Chinh dien'),
        (POSE_LEFT, 'Quay trai'),
        (POSE_RIGHT, 'Quay phai'),
        (POSE_UP, 'Quay len'),
        (POSE_DOWN, 'Quay xuong'),
    ]

    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE,
        related_name='face_embeddings',
        verbose_name='Nhan vien',
    )
    embedding = models.BinaryField(
        null=True,
        blank=True,
        verbose_name='Vector khuon mat binary',
        help_text='Du lieu vector khuon mat luu truc tiep trong PostgreSQL',
    )
    registration = models.ForeignKey(
        'FaceRegistration',
        on_delete=models.SET_NULL,
        related_name='embeddings',
        null=True,
        blank=True,
        verbose_name='Lan dang ky',
    )
    pose = models.CharField(
        max_length=20,
        choices=POSE_CHOICES,
        default=POSE_FRONT,
        verbose_name='Goc chup',
    )
    employee_code_snapshot = models.CharField(
        max_length=50,
        blank=True,
        default='',
        verbose_name='Ma nhan vien tai thoi diem dang ky',
    )
    employee_name_snapshot = models.CharField(
        max_length=255,
        blank=True,
        default='',
        verbose_name='Ten nhan vien tai thoi diem dang ky',
    )
    cloudinary_url = models.URLField(
        max_length=1000,
        null=True,
        blank=True,
        verbose_name='Link anh khuon mat Cloudinary',
    )
    image = models.ImageField(
        upload_to='face_crops/%Y/%m/%d/',
        null=True,
        blank=True,
        verbose_name='Anh khuon mat',
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngay tao')

    class Meta:
        db_table = 'face_embeddings'
        verbose_name = 'Du lieu khuon mat'
        verbose_name_plural = 'Du lieu khuon mat'

    def __str__(self):
        return f"FaceEmbedding for {self.employee}"


class FaceLog(models.Model):
    employee = models.ForeignKey(
        'employees.Employee',
        on_delete=models.CASCADE,
        related_name='face_logs',
        verbose_name='Nhan vien',
    )
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name='Thoi gian quet')
    confidence = models.FloatField(default=0.0, verbose_name='Do tin cay')
    image = models.ImageField(upload_to='face_logs/%Y/%m/%d/', null=True, blank=True, verbose_name='Anh')

    class Meta:
        db_table = 'face_logs'
        ordering = ['-timestamp']
        verbose_name = 'Lich su nhan dien'
        verbose_name_plural = 'Lich su nhan dien'

    def __str__(self):
        return f"{self.employee} - {self.timestamp}"


class FaceRegistration(models.Model):
    user_id = models.CharField(max_length=255, verbose_name='ID nguoi dung')
    employee = models.ForeignKey(
        Employee,
        on_delete=models.SET_NULL,
        related_name='face_registrations',
        null=True,
        blank=True,
        verbose_name='Nhan vien',
    )
    employee_code = models.CharField(max_length=50, blank=True, default='', verbose_name='Ma nhan vien')
    employee_name = models.CharField(max_length=255, blank=True, default='', verbose_name='Ten nhan vien')
    image_count = models.IntegerField(default=0, verbose_name='So anh da dang ky')
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Dang cho'),
            ('completed', 'Hoan tat'),
            ('failed', 'That bai'),
        ],
        default='pending',
        verbose_name='Trang thai',
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Thoi gian tao')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Thoi gian cap nhat')

    class Meta:
        db_table = 'face_registrations'
        ordering = ['-created_at']
        verbose_name = 'Dang ky khuon mat'
        verbose_name_plural = 'Dang ky khuon mat'

    def __str__(self):
        return f"FaceRegistration for {self.user_id} ({self.status})"
