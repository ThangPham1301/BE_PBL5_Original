from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.core.files.base import ContentFile
from datetime import timedelta

from apps.employees.models import Employee
from apps.attendance.models import AttendanceLog
from .models import FaceLog
from .services import FaceRecognitionService

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

