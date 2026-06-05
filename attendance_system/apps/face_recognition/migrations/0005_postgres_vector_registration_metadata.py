from django.db import migrations, models
import django.contrib.postgres.fields
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('face_recognition', '0004_faceembedding_cloudinary_url_and_nullable_embedding'),
    ]

    operations = [
        migrations.AddField(
            model_name='faceregistration',
            name='employee',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='face_registrations', to='employees.employee', verbose_name='Nhan vien'),
        ),
        migrations.AddField(
            model_name='faceregistration',
            name='employee_code',
            field=models.CharField(blank=True, default='', max_length=50, verbose_name='Ma nhan vien'),
        ),
        migrations.AddField(
            model_name='faceregistration',
            name='employee_name',
            field=models.CharField(blank=True, default='', max_length=255, verbose_name='Ten nhan vien'),
        ),
        migrations.AddField(
            model_name='faceembedding',
            name='embedding_vector',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), blank=True, null=True, size=512, verbose_name='Vector khuon mat PostgreSQL array'),
        ),
        migrations.AddField(
            model_name='faceembedding',
            name='employee_code_snapshot',
            field=models.CharField(blank=True, default='', max_length=50, verbose_name='Ma nhan vien tai thoi diem dang ky'),
        ),
        migrations.AddField(
            model_name='faceembedding',
            name='employee_name_snapshot',
            field=models.CharField(blank=True, default='', max_length=255, verbose_name='Ten nhan vien tai thoi diem dang ky'),
        ),
        migrations.AddField(
            model_name='faceembedding',
            name='pose',
            field=models.CharField(choices=[('front', 'Chinh dien'), ('left', 'Quay trai'), ('right', 'Quay phai'), ('up', 'Quay len'), ('down', 'Quay xuong')], default='front', max_length=20, verbose_name='Goc chup'),
        ),
        migrations.AddField(
            model_name='faceembedding',
            name='registration',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='embeddings', to='face_recognition.faceregistration', verbose_name='Lan dang ky'),
        ),
    ]
