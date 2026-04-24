from rest_framework import serializers
from .models import FaceRegistration


class FaceRegistrationSerializer(serializers.ModelSerializer):
    """
    Serializer for Face Registration
    """
    class Meta:
        model = FaceRegistration
        fields = [
            'id',
            'user_id',
            'image_count',
            'status',
            'created_at',
            'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class FaceRegisterRequestSerializer(serializers.Serializer):
    """
    Serializer for Face Registration Request
    Receives user_id and list of base64 images
    """
    user_id = serializers.CharField(max_length=255)
    images = serializers.ListField(
        child=serializers.CharField(),
        min_length=1,
        max_length=10
    )

    def validate_images(self, value):
        """
        Validate that images list is not empty
        """
        if not value or len(value) == 0:
            raise serializers.ValidationError('Phải cung cấp ít nhất 1 ảnh')
        
        if len(value) > 10:
            raise serializers.ValidationError('Tối đa 10 ảnh')
        
        return value


class FaceValidateRequestSerializer(serializers.Serializer):
    """Serializer for pre-validating a single base64 face image."""
    image = serializers.CharField()

    def validate_image(self, value):
        if not value:
            raise serializers.ValidationError('Thiếu dữ liệu ảnh')
        return value
