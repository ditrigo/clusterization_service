from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DatasetViewSet, ClusteringJobViewSet, PresetListView

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'clustering-jobs', ClusteringJobViewSet, basename='clusteringjob')

urlpatterns = [
    path('', include(router.urls)),
    path('presets/', PresetListView.as_view(), name='presets'),
]
