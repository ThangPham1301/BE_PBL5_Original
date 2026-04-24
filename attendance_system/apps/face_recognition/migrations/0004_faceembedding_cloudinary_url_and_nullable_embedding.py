from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('face_recognition', '0003_faceregistration'),
    ]

    operations = [
        migrations.AddField(
            model_name='faceembedding',
            name='cloudinary_url',
            field=models.URLField(blank=True, max_length=1000, null=True, verbose_name='Link ảnh khuôn mặt (Cloudinary)'),
        ),
        migrations.AlterField(
            model_name='faceembedding',
            name='embedding',
            field=models.BinaryField(blank=True, help_text='Dữ liệu vector khuôn mặt được mã hóa', null=True, verbose_name='Vector đặc trưng khuôn mặt'),
        ),
    ]
