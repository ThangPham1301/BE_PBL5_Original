from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.core.files.base import ContentFile
from datetime import timedelta
import base64
import json
import numpy as np
import cv2
from django.db import transaction

from apps.employees.models import Employee
from apps.attendance.models import AttendanceLog
from .models import FaceEmbedding, FaceLog, FaceRegistration
from .serializers import FaceRegisterRequestSerializer, FaceValidateRequestSerializer
from .services import FaceRecognitionService


class FaceValidateAPIView(APIView):
    """Validate a single base64 image and report whether the face is clear enough."""
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = FaceValidateRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    'success': False,
                    'is_clear': False,
                    'message': 'Dữ liệu không hợp lệ',
                    'errors': serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_data = serializer.validated_data['image']

        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return Response(
                    {
                        'success': True,
                        'is_clear': False,
                        'message': 'Ảnh không hợp lệ, vui lòng giữ máy ổn định.',
                    }
                )

            service = FaceRecognitionService()
            if service.detector is None:
                service.initialize()

            if service.detector is None:
                return Response(
                    {
                        'success': False,
                        'is_clear': False,
                        'message': 'Bộ nhận diện khuôn mặt chưa sẵn sàng',
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            detections = service.detector.detect(frame, conf_threshold=0.45)
            if not detections:
                return Response(
                    {
                        'success': True,
                        'is_clear': False,
                        'message': 'Chưa thấy khuôn mặt rõ trong khung.',
                    }
                )

            best_face = max(detections, key=lambda d: d.w * d.h)
            face_area = float(best_face.w * best_face.h)
            frame_area = float(frame.shape[0] * frame.shape[1])
            area_ratio = face_area / frame_area if frame_area > 0 else 0.0

            x1, y1, x2, y2 = best_face.as_tuple()
            face_crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            if face_crop.size == 0:
                return Response(
                    {
                        'success': True,
                        'is_clear': False,
                        'message': 'Không crop được khuôn mặt, vui lòng thử lại.',
                    }
                )

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            min_face_area_ratio = 0.08
            min_blur_score = 60.0

            if area_ratio < min_face_area_ratio:
                return Response(
                    {
                        'success': True,
                        'is_clear': False,
                        'message': 'Khuôn mặt còn nhỏ, vui lòng đưa camera lại gần hơn.',
                        'metrics': {
                            'face_area_ratio': round(area_ratio, 4),
                            'blur_score': round(blur_score, 2),
                        },
                    }
                )

            if blur_score < min_blur_score:
                return Response(
                    {
                        'success': True,
                        'is_clear': False,
                        'message': 'Ảnh bị mờ, vui lòng giữ yên vài giây và đủ sáng.',
                        'metrics': {
                            'face_area_ratio': round(area_ratio, 4),
                            'blur_score': round(blur_score, 2),
                        },
                    }
                )

            return Response(
                {
                    'success': True,
                    'is_clear': True,
                    'message': 'Khuôn mặt rõ, có thể chụp.',
                    'metrics': {
                        'face_area_ratio': round(area_ratio, 4),
                        'blur_score': round(blur_score, 2),
                    },
                }
            )

        except Exception as e:
            return Response(
                {
                    'success': False,
                    'is_clear': False,
                    'message': f'Lỗi kiểm tra ảnh: {str(e)}',
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

class RecognizeAPIView(APIView):
    """
    API Endpoint for Face Recognition.
    Accepts an image file.
    Returns the identified employee or unknown.
    """
    permission_classes = [AllowAny]
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'file' not in request.data:
             return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        file_obj = request.data['file']
        
        try:
            image_bytes = file_obj.read()
            service = FaceRecognitionService()
            found_face, employee_id, confidence = service.process_image(image_bytes)
            
            if not found_face:
                 return Response({
                     'success': False,
                     'message': 'No face detected in the image.'
                 }, status=status.HTTP_400_BAD_REQUEST)
                 
            if employee_id:
                 employee = Employee.objects.get(pk=employee_id)
                 
                 # 1. Create Face Recognition Log (History)
                 # Save image to log (optional, but good for history)
                 log_entry = FaceLog.objects.create(
                     employee=employee,
                     confidence=confidence,
                     image=ContentFile(image_bytes, name=f"{employee.id}_{timezone.now().timestamp()}.jpg")
                 )
                 
                 # 2. Process Attendance (Debounce based on last check-out)
                 today = timezone.localdate()
                 now = timezone.now()
                 
                 attendance_log, created = AttendanceLog.objects.get_or_create(
                     employee=employee,
                     date=today,
                     defaults={
                         'check_in': now,
                         'status': AttendanceLog.Status.PRESENT
                     }
                 )
                 
                 attendance_msg = "Đã điểm danh"
                 
                 if created:
                     attendance_msg = "Check-in thành công (Mới)"
                 else:
                     updated_fields = []
                     # Helper to check if we should update based on debounce time
                     # Debounce: 5 minutes
                     DEBOUNCE_TIME = timedelta(minutes=5)
                     
                     # 1. If check-in is missing (rare case), fill it
                     if not attendance_log.check_in:
                         attendance_log.check_in = now
                         updated_fields.append('check_in')
                         attendance_msg = "Cập nhật giờ vào"
                     
                     # 2. Update Check-out (Last In - Last Out strategy)
                     # Only update if check_out is None OR it's been > 5 mins since last check_out
                     last_checkout = attendance_log.check_out
                     if not last_checkout or (now - last_checkout > DEBOUNCE_TIME):
                         attendance_log.check_out = now
                         updated_fields.append('check_out')
                         attendance_msg = "Cập nhật giờ ra"
                     else:
                         attendance_msg = "Đã ghi nhận (Debounce)"

                     if updated_fields:
                         attendance_log.save(update_fields=updated_fields)

                 return Response({
                     'success': True,
                     'identified': True,
                     'employee_id': employee.id,
                     'name': str(employee),
                     'confidence': confidence,
                     'attendance_message': attendance_msg
                 })
            else:
                 return Response({
                     'success': True,
                     'identified': False,
                     'message': 'Face detected but not recognized.',
                     'confidence': confidence
                 })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class EnrollAPIView(APIView):
    """
    API Endpoint to enroll a face for an employee.
    """
    permission_classes = [AllowAny]
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        employee_id = request.data.get('employee_id')
        if not employee_id:
            return Response({'error': 'employee_id required'}, status=status.HTTP_400_BAD_REQUEST)
            
        if 'file' not in request.data:
             return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
             
        try:
            file_obj = request.data['file']
            image_bytes = file_obj.read()
            
            service = FaceRecognitionService()
            service.enroll_face(image_bytes, employee_id)
            
            return Response({'success': True, 'message': f'Face enrolled for employee {employee_id}'})
            
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FaceRegistrationAPIView(APIView):
    """
    API Endpoint for Face Registration
    
    POST /api/face/register/
    
    Request:
    {
        "user_id": "123",
        "images": ["base64_image_1", "base64_image_2", ...]
    }
    
    Response:
    {
        "success": true,
        "message": "Đã đăng ký khuôn mặt thành công",
        "registration_id": 1,
        "image_count": 5
    }
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = FaceRegisterRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                {
                    'success': False,
                    'message': 'Dữ liệu không hợp lệ',
                    'errors': serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        user_id = serializer.validated_data['user_id']
        images = serializer.validated_data['images']
        service = FaceRecognitionService()

        try:
            try:
                employee = Employee.objects.get(pk=user_id)
            except Employee.DoesNotExist:
                employee = Employee.objects.get(employee_id=user_id)
        except Employee.DoesNotExist:
            return Response(
                {
                    'success': False,
                    'message': f'Không tìm thấy nhân viên cho user_id={user_id}'
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            saved_count = 0
            with transaction.atomic():
                # Xóa tất cả dữ liệu khuôn mặt cũ của nhân viên này (chỉ giữ 1 bộ duy nhất)
                FaceEmbedding.objects.filter(employee=employee).delete()
                FaceRegistration.objects.filter(user_id=str(user_id)).delete()
                print(f"[FACE REGISTRATION] Đã xóa dữ liệu khuôn mặt cũ cho user {user_id}")
                
                for idx, image_data in enumerate(images):
                    try:
                        if ',' in image_data:
                            image_data = image_data.split(',')[1]

                        image_bytes = base64.b64decode(image_data)
                        image_url = service.upload_face_image_to_cloudinary(
                            image_bytes=image_bytes,
                            employee_id=employee.id,
                        )

                        FaceEmbedding.objects.create(
                            employee=employee,
                            cloudinary_url=image_url,
                        )
                        saved_count += 1
                        print(f"[FACE REGISTRATION] Ảnh {idx + 1} cho user {user_id}: uploaded to Cloudinary")

                    except Exception as e:
                        print(f"[FACE REGISTRATION ERROR] Không thể xử lý ảnh {idx + 1}: {str(e)}")
                        continue

                if saved_count == 0:
                    raise ValueError('Không có ảnh hợp lệ để upload')

                registration = FaceRegistration.objects.create(
                    user_id=user_id,
                    image_count=saved_count,
                    status='completed'
                )

            return Response(
                {
                    'success': True,
                    'message': 'Đã đăng ký khuôn mặt thành công',
                    'registration_id': registration.id,
                    'image_count': registration.image_count,
                    'status': registration.status
                },
                status=status.HTTP_201_CREATED
            )

        except ValueError as e:
            return Response(
                {
                    'success': False,
                    'message': str(e)
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {
                    'success': False,
                    'message': f'Lỗi xử lý đăng ký: {str(e)}'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

