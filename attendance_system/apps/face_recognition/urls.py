from django.urls import path
from .views import RecognizeAPIView, EnrollAPIView, FaceRegistrationAPIView, FaceValidateAPIView

urlpatterns = [
    path('recognize/', RecognizeAPIView.as_view(), name='recognize'),
    path('enroll/', EnrollAPIView.as_view(), name='enroll'),
    path('validate/', FaceValidateAPIView.as_view(), name='face-validate'),
    path('register/', FaceRegistrationAPIView.as_view(), name='face-register'),
]
