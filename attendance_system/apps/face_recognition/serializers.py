from rest_framework import serializers

from .models import FaceRegistration


DEFAULT_FACE_POSES = ['front', 'left', 'right', 'up', 'down']


class FaceRegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceRegistration
        fields = [
            'id',
            'user_id',
            'employee',
            'employee_code',
            'employee_name',
            'image_count',
            'status',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class FaceRegisterRequestSerializer(serializers.Serializer):
    user_id = serializers.CharField(max_length=255)
    images = serializers.ListField(
        child=serializers.CharField(),
        min_length=5,
        max_length=5,
    )
    poses = serializers.ListField(
        child=serializers.ChoiceField(choices=DEFAULT_FACE_POSES),
        required=False,
        min_length=5,
        max_length=5,
    )

    def validate(self, attrs):
        attrs = super().validate(attrs)
        attrs['poses'] = attrs.get('poses') or DEFAULT_FACE_POSES
        if len(attrs['images']) != 5:
            raise serializers.ValidationError({'images': 'Phải cung cấp đúng 5 ảnh khuôn mặt.'})
        if len(attrs['poses']) != 5:
            raise serializers.ValidationError({'poses': 'Phải cung cấp đúng 5 góc chụp.'})
        return attrs


class FaceValidateRequestSerializer(serializers.Serializer):
    image = serializers.CharField()
    pose = serializers.ChoiceField(choices=DEFAULT_FACE_POSES, required=False)

    def validate_image(self, value):
        if not value:
            raise serializers.ValidationError('Thiếu dữ liệu ảnh.')
        return value
