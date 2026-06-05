from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.utils import timezone
from django.core.files.base import ContentFile
from datetime import timedelta
import base64
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
                    'can_capture': False,
                    'message': 'Dữ liệu không hợp lệ',
                    'errors': serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_data = serializer.validated_data['image']

        try:
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]

            image_bytes = base64.b64decode(image_data, validate=True)
            service = FaceRecognitionService()
            validation = service.validate_registration_image(
                image_bytes,
                expected_pose=serializer.validated_data.get('pose'),
            )
            return Response(
                {
                    'success': True,
                    'is_clear': validation['is_clear'],
                    'can_capture': validation['is_clear'],
                    'message': validation['message'],
                    'metrics': validation['metrics'],
                    'face_box': validation['face_box'],
                    'landmarks': validation['landmarks'],
                    'frame_size': validation['frame_size'],
                }
            )

        except Exception as e:
            return Response(
                {
                    'success': False,
                    'is_clear': False,
                    'can_capture': False,
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
                employee_name = employee.user.get_full_name() or employee.user.username
                print(
                    f"[FACE CHECK-IN] employee_code={employee.employee_id} "
                    f"employee_name={employee_name} confidence={confidence:.4f}"
                )

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

                attendance_msg = "Da diem danh"

                if created:
                    attendance_msg = "Check-in thanh cong (Moi)"
                else:
                    updated_fields = []
                    DEBOUNCE_TIME = timedelta(minutes=5)

                    if not attendance_log.check_in:
                        attendance_log.check_in = now
                        updated_fields.append('check_in')
                        attendance_msg = "Cap nhat gio vao"

                    last_checkout = attendance_log.check_out
                    if not last_checkout or (now - last_checkout > DEBOUNCE_TIME):
                        attendance_log.check_out = now
                        updated_fields.append('check_out')
                        attendance_msg = "Cap nhat gio ra"
                    else:
                        attendance_msg = "Da ghi nhan (Debounce)"

                    if updated_fields:
                        attendance_log.save(update_fields=updated_fields)

                return Response({
                    'success': True,
                    'identified': True,
                    'employee_id': employee.id,
                    'employee_code': employee.employee_id,
                    'employee_name': employee_name,
                    'name': str(employee),
                    'confidence': confidence,
                    'attendance_message': attendance_msg,
                })
            else:
                return Response({
                    'success': True,
                    'identified': False,
                    'message': 'Face detected but not recognized.',
                    'confidence': confidence,
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


class FaceRegistrationAPIView(APIView):
    """Register 5 face angles and persist embeddings with employee metadata."""

    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = FaceRegisterRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    'success': False,
                    'message': 'Dữ liệu không hợp lệ',
                    'errors': serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        user_id = serializer.validated_data['user_id']
        images = serializer.validated_data['images']
        poses = serializer.validated_data['poses']
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
                    'message': f'Không tìm thấy nhân viên cho user_id={user_id}',
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        employee_name = employee.user.get_full_name() or employee.user.username

        try:
            saved_count = 0
            with transaction.atomic():
                FaceEmbedding.objects.filter(employee=employee).delete()
                FaceRegistration.objects.filter(user_id=str(user_id)).delete()

                registration = FaceRegistration.objects.create(
                    user_id=user_id,
                    employee=employee,
                    employee_code=employee.employee_id,
                    employee_name=employee_name,
                    image_count=0,
                    status='pending',
                )

                for idx, image_data in enumerate(images):
                    try:
                        if ',' in image_data:
                            image_data = image_data.split(',', 1)[1]

                        image_bytes = base64.b64decode(image_data, validate=True)
                        embedding = service.extract_embedding_from_bytes(
                            image_bytes,
                            expected_pose=poses[idx],
                            require_pose=True,
                        )
                        embedding = service._normalize_embedding(embedding)

                        image_url = None
                        try:
                            image_url = service.upload_face_image_to_cloudinary(
                                image_bytes=image_bytes,
                                employee_id=employee.id,
                                expected_pose=poses[idx],
                            )
                        except Exception as upload_error:
                            print(f"[FACE REGISTRATION WARN] Cloudinary skipped for image {idx + 1}: {upload_error}")

                        FaceEmbedding.objects.create(
                            employee=employee,
                            registration=registration,
                            pose=poses[idx],
                            embedding=embedding.tobytes(),
                            employee_code_snapshot=employee.employee_id,
                            employee_name_snapshot=employee_name,
                            cloudinary_url=image_url,
                        )
                        saved_count += 1
                        print(f"[FACE REGISTRATION] Saved embedding {idx + 1}/5 pose={poses[idx]} employee={employee.employee_id}")

                    except Exception as exc:
                        print(f"[FACE REGISTRATION ERROR] Image {idx + 1} pose={poses[idx]} failed: {exc}")
                        continue

                if saved_count != 5:
                    raise ValueError(f'Chỉ lưu được {saved_count}/5 ảnh hợp lệ. Vui lòng đăng ký lại.')

                registration.image_count = saved_count
                registration.status = 'completed'
                registration.save(update_fields=['image_count', 'status', 'updated_at'])
                service.invalidate_cache()

            return Response(
                {
                    'success': True,
                    'message': 'Đã đăng ký khuôn mặt và lưu embedding thành công',
                    'registration_id': registration.id,
                    'employee_id': employee.id,
                    'employee_code': employee.employee_id,
                    'employee_name': employee_name,
                    'image_count': registration.image_count,
                    'poses': poses,
                    'status': registration.status,
                },
                status=status.HTTP_201_CREATED,
            )

        except ValueError as exc:
            return Response(
                {
                    'success': False,
                    'message': str(exc),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            return Response(
                {
                    'success': False,
                    'message': f'Lỗi xử lý đăng ký: {exc}',
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
