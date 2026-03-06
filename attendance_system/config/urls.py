from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('apps.accounts.urls')),
    path('api/', include('apps.employees.urls')),
    path('api/', include('apps.shifts.urls')),
    path('api/', include('apps.attendance.urls')),
    path('api/', include('apps.leaves.urls')),
    path('api/', include('apps.reports.urls')),
    path('api/face/', include('apps.face_recognition.urls')),
]
