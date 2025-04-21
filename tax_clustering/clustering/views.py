import matplotlib
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
matplotlib.use('Agg')
import pandas as pd
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action, api_view
from rest_framework.views import APIView
import shap
from .models import Dataset, ClusteringJob
from .serializers import DatasetSerializer, ClusteringJobSerializer
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser
from clustering.modules import risk
import numpy as np
from clustering.modules.risk import generate_shap_plot, compute_risk_score
import os
import logging
from django.conf import settings
from .presets import PRESETS
from datetime import datetime
import pickle

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
            job.status = 'Completed'
            job.completed_at = datetime.now()
            job.save()
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
        
        
class PresetListView(APIView):
    def get(self, request):
        preset_name = request.query_params.get('name', None)
        if preset_name:
            preset = next((p for p in PRESETS if p['name'] == preset_name), None)
            if preset:
                return Response(preset, status=status.HTTP_200_OK)
            else:
                return Response({"detail": "Пресет не найден."}, status=status.HTTP_404_NOT_FOUND)
        return Response(PRESETS, status=status.HTTP_200_OK)
    
    
class RiskAnalysisView(APIView):
    def get(self, request, job_id):
        job = ClusteringJob.objects.get(id=job_id)
        path = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id), 'clustering.csv')
        df = pd.read_csv(path)
        df_data = df.drop(columns=['Cluster'], errors='ignore')
        raw = df_data.sum(axis=1)
        lo, hi = raw.min(), raw.max()
        if hi > lo:
            norm = (raw - lo) / (hi - lo)
        else:
            norm = pd.Series(0, index=raw.index)
        df['risk_score'] = norm
        df['risk_color'] = df['risk_score'].apply(
            lambda v: 'green' if v < .50 else ('yellow' if v < .75 else 'red')
        )
        return Response({'data': df.to_dict(orient='records')}, status=200)
        
        

class RiskShapExplanationView(APIView):
    def get(self, request, job_id, record_index):
        try:
            job = ClusteringJob.objects.get(id=job_id)
        except ClusteringJob.DoesNotExist:
            return Response({"detail": "Job не найдено"}, status=404)
        clustering_file = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id), 'clustering.csv')
        if not os.path.exists(clustering_file):
            return Response({"detail": "clustering.csv не найден. Выполните кластеризацию."}, status=404)
        df = pd.read_csv(clustering_file)
        try:
            idx = int(record_index)
        except ValueError:
            return Response({"detail": "record_index должен быть целым"}, status=400)
        if idx < 0 or idx >= len(df):
            return Response({"detail": f"Index {idx} вне диапазона (0–{len(df)-1})"}, status=400)
        data_row = df.drop(columns=['Cluster'], errors='ignore').iloc[[idx]]
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'risk_model.pkl')
        if not os.path.exists(model_path):
            X = df.drop(columns=['Cluster'], errors='ignore')
            y = X.apply(lambda row: risk.compute_risk_score(row.to_frame().T), axis=1)
            model = DecisionTreeRegressor(random_state=42).fit(X, y)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(data_row)
        shap_dir = os.path.join(settings.MEDIA_ROOT, 'risk_explanations', str(job.id))
        os.makedirs(shap_dir, exist_ok=True)
        shap_filename = f"shap_{idx}.png"
        shap_path = os.path.join(shap_dir, shap_filename)
        plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        shap_url = os.path.join(settings.MEDIA_URL, 'risk_explanations', str(job.id), shap_filename)
        return Response({"shap_url": shap_url}, status=200)