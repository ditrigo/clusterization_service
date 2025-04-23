from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (ClusteringJobViewSet, DatasetViewSet, PresetListView,
                    RiskAnalysisView, RiskShapExplanationView)

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'clustering-jobs', ClusteringJobViewSet, basename='clusteringjob')

urlpatterns = [
    path('', include(router.urls)),
    path('presets/', PresetListView.as_view(), name='presets'),
    path('risk-analysis/<uuid:job_id>/', RiskAnalysisView.as_view(), name='risk_analysis'),
    path('risk-analysis/<uuid:job_id>/<int:record_index>/', RiskShapExplanationView.as_view(), name='risk_shap_explanation'),
]
