from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import LeaveTypeViewSet, LeaveRequestViewSet

router = DefaultRouter()
router.register(r'leave-types', LeaveTypeViewSet)
router.register(r'leaves', LeaveRequestViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
