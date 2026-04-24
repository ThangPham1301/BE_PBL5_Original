from django.db import models
from apps.employees.models import Employee

class FaceEmbedding(models.Model):
    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE,
        related_name='face_embeddings',
        verbose_name='Nhân viên'
    )
    embedding = models.BinaryField(
        verbose_name='Vector đặc trưng khuôn mặt',
        help_text='Dữ liệu vector khuôn mặt được mã hóa',
        null=True,
        blank=True,
    )
    cloudinary_url = models.URLField(
        max_length=1000,
        null=True,
        blank=True,
        verbose_name='Link ảnh khuôn mặt (Cloudinary)'
    )
    # Lưu ảnh gốc (crop) để tham chiếu/kiểm tra sau này nếu cần
    image = models.ImageField(
        upload_to='face_crops/%Y/%m/%d/',
        null=True,
        blank=True,
        verbose_name='Ảnh khuôn mặt'
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Ngày tạo')

    class Meta:
        db_table = 'face_embeddings'
        verbose_name='Dữ liệu khuôn mặt'
        verbose_name_plural='Dữ liệu khuôn mặt'

    def __str__(self):
        return f"FaceEmbedding for {self.employee}"


class FaceLog(models.Model):
    employee = models.ForeignKey(
        'employees.Employee', 
        on_delete=models.CASCADE, 
        related_name='face_logs',
        verbose_name='Nhân viên'
    )
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name='Thời gian quét')
    confidence = models.FloatField(default=0.0, verbose_name='Độ tin cậy')
    image = models.ImageField(upload_to='face_logs/%Y/%m/%d/', null=True, blank=True, verbose_name='Ảnh')

    class Meta:
        db_table = 'face_logs'
        ordering = ['-timestamp']
        verbose_name = 'Lịch sử nhận diện'
        verbose_name_plural = 'Lịch sử nhận diện'

    def __str__(self):
        return f"{self.employee} - {self.timestamp}"


class FaceRegistration(models.Model):
    """
    Lưu thông tin quá trình đăng ký khuôn mặt
    """
    user_id = models.CharField(
        max_length=255,
        verbose_name='ID Người dùng'
    )
    image_count = models.IntegerField(
        default=0,
        verbose_name='Số ảnh đã đăng ký'
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Đang chờ'),
            ('completed', 'Hoàn tất'),
            ('failed', 'Thất bại'),
        ],
        default='pending',
        verbose_name='Trạng thái'
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Thời gian tạo'
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name='Thời gian cập nhật'
    )

    class Meta:
        db_table = 'face_registrations'
        ordering = ['-created_at']
        verbose_name = 'Đăng ký khuôn mặt'
        verbose_name_plural = 'Đăng ký khuôn mặt'

    def __str__(self):
        return f"FaceRegistration for {self.user_id} ({self.status})"


