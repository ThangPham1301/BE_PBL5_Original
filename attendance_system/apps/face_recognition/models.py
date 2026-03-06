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
        help_text='Dữ liệu vector khuôn mặt được mã hóa'
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
        verbose_name = 'Dữ liệu khuôn mặt'
        verbose_name_plural = 'Dữ liệu khuôn mặt'

    def __str__(self):
        return f"Face of {self.employee} ({self.created_at})"

