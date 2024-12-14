from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('clustering.urls')),
    path('', include('frontend.urls')),  # Добавьте эту строку для фронтенда
]

# Добавляем маршруты для обслуживания медиа-файлов только в режиме DEBUG
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
