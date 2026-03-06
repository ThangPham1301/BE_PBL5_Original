from django.urls import path
from .views import RecognizeAPIView, EnrollAPIView

urlpatterns = [
    path('recognize/', RecognizeAPIView.as_view(), name='recognize'),
    path('enroll/', EnrollAPIView.as_view(), name='enroll'),
]
