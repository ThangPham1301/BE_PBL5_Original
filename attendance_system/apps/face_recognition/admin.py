from django.contrib import admin
from .models import FaceEmbedding, FaceLog, FaceRegistration

@admin.register(FaceEmbedding)
class FaceEmbeddingAdmin(admin.ModelAdmin):
	list_display = ['id', 'employee', 'cloudinary_url', 'created_at']
	list_filter = ['created_at']
	search_fields = ['employee__name']

@admin.register(FaceLog)
class FaceLogAdmin(admin.ModelAdmin):
	list_display = ['id', 'employee', 'timestamp', 'confidence']
	list_filter = ['timestamp']
	search_fields = ['employee__name']

@admin.register(FaceRegistration)
class FaceRegistrationAdmin(admin.ModelAdmin):
	list_display = ['id', 'user_id', 'image_count', 'status', 'created_at']
	list_filter = ['status', 'created_at']
	search_fields = ['user_id']
	readonly_fields = ['id', 'created_at', 'updated_at']
    
	fieldsets = (
		('Thông tin cơ bản', {
			'fields': ('user_id', 'status')
		}),
		('Dữ liệu', {
			'fields': ('image_count',)
		}),
		('Thời gian', {
			'fields': ('created_at', 'updated_at')
		}),
	)
