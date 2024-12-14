from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action, api_view
from .models import Dataset, ClusteringJob
from .serializers import DatasetSerializer, ClusteringJobSerializer
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser
import os
import logging
from django.conf import settings

# Импорт функций обработки
from .pipeline import (
    execute_preprocessing,
    execute_feature_selection,
    execute_dimensionality_reduction,
    execute_clustering,
    execute_metrics,
    execute_visualization,
    execute_all_steps
)

logger = logging.getLogger(__name__)

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all().order_by('-uploaded_at')
    serializer_class = DatasetSerializer
    parser_classes = [MultiPartParser, FormParser]

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context

class ClusteringJobViewSet(viewsets.ModelViewSet):
    queryset = ClusteringJob.objects.all().order_by('-created_at')
    serializer_class = ClusteringJobSerializer

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save()
        # Инициализация промежуточной директории
        intermediate_dir = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id))
        os.makedirs(intermediate_dir, exist_ok=True)
        # Возвращаем ответ
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    @action(detail=True, methods=['post'])
    def preprocess(self, request, pk=None):
        job = self.get_object()
        if job.preprocessing_completed:
            return Response({"detail": "Preprocessing already completed."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            execute_preprocessing(job)
            serializer = self.get_serializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            job.status = 'Failed'
            job.save()
            logger.error(f"Error during preprocessing for job {job.id}: {e}", exc_info=True)
            return Response({"detail": "Error during preprocessing."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def feature_selection(self, request, pk=None):
        job = self.get_object()
        if not job.preprocessing_completed:
            return Response({"detail": "Preprocessing not completed."}, status=status.HTTP_400_BAD_REQUEST)
        if job.feature_selection_completed:
            return Response({"detail": "Feature selection already completed."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            execute_feature_selection(job)
            serializer = self.get_serializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            job.status = 'Failed'
            job.save()
            logger.error(f"Error during feature selection for job {job.id}: {e}", exc_info=True)
            return Response({"detail": "Error during feature selection."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def dimensionality_reduction(self, request, pk=None):
        job = self.get_object()
        if not job.feature_selection_completed:
            return Response({"detail": "Feature selection not completed."}, status=status.HTTP_400_BAD_REQUEST)
        if job.dimensionality_reduction_completed:
            return Response({"detail": "Dimensionality reduction already completed."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            execute_dimensionality_reduction(job)
            serializer = self.get_serializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            job.status = 'Failed'
            job.save()
            logger.error(f"Error during dimensionality reduction for job {job.id}: {e}", exc_info=True)
            return Response({"detail": "Error during dimensionality reduction."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def clustering(self, request, pk=None):
        job = self.get_object()
        if not job.dimensionality_reduction_completed:
            return Response({"detail": "Dimensionality reduction not completed."}, status=status.HTTP_400_BAD_REQUEST)
        if job.clustering_completed:
            return Response({"detail": "Clustering already completed."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            execute_clustering(job)
            serializer = self.get_serializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            job.status = 'Failed'
            job.save()
            logger.error(f"Error during clustering for job {job.id}: {e}", exc_info=True)
            return Response({"detail": "Error during clustering."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def metrics(self, request, pk=None):
        job = self.get_object()
        if not job.clustering_completed:
            return Response({"detail": "Clustering not completed."}, status=status.HTTP_400_BAD_REQUEST)
        if job.metrics_completed:
            return Response({"detail": "Metrics already calculated."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            execute_metrics(job)
            serializer = self.get_serializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            job.status = 'Failed'
            job.save()
            logger.error(f"Error calculating metrics for job {job.id}: {e}", exc_info=True)
            return Response({"detail": "Error during metrics calculation."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def visualization(self, request, pk=None):
        job = self.get_object()
        if not job.metrics_completed:
            return Response({"detail": "Metrics calculation not completed."}, status=status.HTTP_400_BAD_REQUEST)
        if job.visualization_completed:
            return Response({"detail": "Visualization already created."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            execute_visualization(job)
            serializer = self.get_serializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            job.status = 'Failed'
            job.save()
            logger.error(f"Error creating visualizations for job {job.id}: {e}", exc_info=True)
            return Response({"detail": "Error during visualization creation."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def execute_all(self, request, pk=None):
        """
        Выполнить все стадии обработки последовательно.
        """
        job = self.get_object()
        try:
            if not job.preprocessing_completed:
                execute_preprocessing(job)
            if not job.feature_selection_completed:
                execute_feature_selection(job)
            if not job.dimensionality_reduction_completed:
                execute_dimensionality_reduction(job)
            if not job.clustering_completed:
                execute_clustering(job)
            if not job.metrics_completed:
                execute_metrics(job)
            if not job.visualization_completed:
                execute_visualization(job)

            # Обновление статуса задания
            job.status = 'Completed'
            # job.completed_at = pd.Timestamp.now()
            job.save()

            serializer = self.get_serializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            job.status = 'Failed'
            job.save()
            logger.error(f"Error during executing all steps for job {job.id}: {e}", exc_info=True)
            return Response({"detail": "Error during executing all steps."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    @action(detail=True, methods=['post'], url_path='execute_all_steps')
    def execute_all_steps_action(self, request, pk=None):
        try:
            job = self.get_object()
            execute_all_steps(job)
            serializer = self.get_serializer(job)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)