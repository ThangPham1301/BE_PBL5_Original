from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import OvertimeRequestViewSet


router = DefaultRouter()
router.register(
    r'overtime-requests',
    OvertimeRequestViewSet,
    basename='overtime-request',
)

urlpatterns = [
    path('', include(router.urls)),
]
