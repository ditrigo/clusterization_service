# frontend/urls.py

from django.urls import path
from . import views

app_name = 'frontend'

urlpatterns = [
    path('', views.home, name='home'),
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.upload_dataset, name='upload_dataset'),
    path('clustering-jobs/', views.job_list, name='job_list'),
    path('clustering-jobs/create/', views.create_job, name='create_job'),
    path('clustering-jobs/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<uuid:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/<uuid:job_id>/execute_all_steps/', views.execute_all_steps, name='execute_all_steps'),
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/<uuid:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('risk-analysis/<uuid:job_id>/', views.risk_analysis, name='risk_analysis'),
]
