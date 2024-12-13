from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action, api_view
from .models import Dataset, ClusteringJob
from .serializers import DatasetSerializer, ClusteringJobSerializer, AvailableStagesSerializer
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser
import os
import logging

# Импорт необходимых функций и модулей
from .modules.preprocessing import (
    handle_missing_values_auto,
    remove_outliers,
    analyze_distributions,
    apply_transformations,
    apply_scalers
)
from .modules.feature_selection import (
    correlation_based_selection,
    kmeans_based_selection,
    mutual_info_selection,
    pca_selection,
    t_sne_selection,
    variance_threshold_selection
)
from .modules.latent_space import (
    autoencoder_selection,
    kernel_pca_selection,
    factor_analysis_selection,
    umap_selection
)
from .modules.clustering_methods import (
    dbscan_clustering,
    gmm_clustering,
    hierarchical_clustering,
    kmeans_clustering,
    optics_clustering,
    spectral_clustering
)
from .modules.metrics import calculate_metrics
from .modules.visualization import visualize_clusters

logger = logging.getLogger(__name__)

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = [MultiPartParser, FormParser]

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context

class ClusteringJobViewSet(viewsets.ModelViewSet):
    queryset = ClusteringJob.objects.all()
    serializer_class = ClusteringJobSerializer

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save()
        # Обработка задания синхронно
        try:
            logger.info(f"Начало обработки задания {job.id}")
            process_clustering_job_sync(job.id)
            logger.info(f"Задание {job.id} успешно завершено.")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            job.status = 'Failed'
            job.save()
            logger.error(f"Ошибка при обработке задания {job.id}: {e}", exc_info=True)
            return Response({"detail": "Error processing job."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['get'])
    def download(self, request, pk=None):
        job = get_object_or_404(ClusteringJob, pk=pk)
        if job.status != 'Completed' or not job.result_file:
            return Response({"detail": "Result not available."}, status=status.HTTP_400_BAD_REQUEST)
        file_path = job.result_file.path
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                response = Response(fh.read(), content_type="text/csv")
                response['Content-Disposition'] = f'attachment; filename={os.path.basename(file_path)}'
                return response
        return Response({"detail": "File not found."}, status=status.HTTP_404_NOT_FOUND)

def process_clustering_job_sync(job_id):
    """
    Синхронная обработка задания кластеризации.
    """
    job = ClusteringJob.objects.get(id=job_id)
    job.status = 'Processing'
    job.save()

    # Загрузка датасета
    dataset_path = job.dataset.file.path
    df = pd.read_csv(dataset_path)

    # Предобработка
    df_cleaned, preprocessing_info = handle_missing_values_auto(df, missing_ratio_threshold=0.4)
    numeric_columns = df_cleaned.select_dtypes(include='number').columns.tolist()
    df_no_outliers, outlier_info = remove_outliers(df_cleaned, numeric_columns=numeric_columns)
    numeric_columns = df_no_outliers.select_dtypes(include='number').columns.tolist()
    column_distributions = analyze_distributions(df_no_outliers, numeric_columns=numeric_columns)
    df_transformed, transform_info = apply_transformations(df_no_outliers, column_distributions)
    df_scaled, scaler_info, df_final = apply_scalers(df_transformed, column_distributions)

    # Отбор параметров (Feature Selection)
    feature_selection_method = job.parameters.feature_selection_method
    feature_selection_params = job.parameters.parameters.get('feature_selection', {})

    if feature_selection_method == 'Correlation':
        threshold = feature_selection_params.get('threshold', 0.6)
        df_selected, selected_features = correlation_based_selection(df_scaled, threshold=threshold)
    elif feature_selection_method == 'KMeans':
        n_clusters = feature_selection_params.get('n_clusters', 3)
        top_n = feature_selection_params.get('top_n', 5)
        df_selected, selected_features = kmeans_based_selection(df_scaled, n_clusters=n_clusters, top_n=top_n)
    elif feature_selection_method == 'Mutual Information':
        n_clusters = feature_selection_params.get('n_clusters', 3)
        top_n = feature_selection_params.get('top_n', 5)
        df_selected, selected_features = mutual_info_selection(df_scaled, n_clusters=n_clusters, top_n=top_n)
    elif feature_selection_method == 'PCA':
        n_components = feature_selection_params.get('n_components', 5)
        df_selected, explained_variance = pca_selection(df_scaled, n_components=n_components)
    elif feature_selection_method == 't-SNE':
        n_components = feature_selection_params.get('n_components', 2)
        perplexity = feature_selection_params.get('perplexity', 30)
        learning_rate = feature_selection_params.get('learning_rate', 200)
        df_selected, tsne = t_sne_selection(
            df_scaled,
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42,
            method='exact'
        )
    elif feature_selection_method == 'Variance Threshold':
        threshold = feature_selection_params.get('threshold', 0.1)
        df_selected, selected_features = variance_threshold_selection(df_scaled, threshold=threshold)
    elif feature_selection_method == 'Autoencoder':
        encoding_dim = feature_selection_params.get('encoding_dim', 10)
        epochs = feature_selection_params.get('epochs', 50)
        batch_size = feature_selection_params.get('batch_size', 32)
        df_selected = autoencoder_selection(df_scaled, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
        selected_features = df_selected.columns.tolist()
    elif feature_selection_method == 'Kernel PCA':
        n_components = feature_selection_params.get('n_components', 10)
        kernel = feature_selection_params.get('kernel', 'rbf')
        gamma = feature_selection_params.get('gamma', None)
        df_selected = kernel_pca_selection(df_scaled, n_components=n_components, kernel=kernel, gamma=gamma)
        selected_features = df_selected.columns.tolist()
    elif feature_selection_method == 'Factor Analysis':
        n_components = feature_selection_params.get('n_components', 10)
        df_selected = factor_analysis_selection(df_scaled, n_components=n_components)
        selected_features = df_selected.columns.tolist()
    elif feature_selection_method == 'UMAP':
        n_neighbors = feature_selection_params.get('n_neighbors', 15)
        min_dist = feature_selection_params.get('min_dist', 0.1)
        n_components = feature_selection_params.get('n_components', 10)
        random_state = feature_selection_params.get('random_state', 42)
        df_selected = umap_selection(
            df_scaled,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state
        )
        selected_features = df_selected.columns.tolist()
    else:
        # Если метод не выбран или неизвестен, используем исходные данные
        df_selected = df_scaled
        selected_features = df_scaled.columns.tolist()

    # Снижение размерности (Dimensionality Reduction)
    dimensionality_reduction_method = job.parameters.dimensionality_reduction_method
    dimensionality_reduction_params = job.parameters.parameters.get('dimensionality_reduction', {})

    if dimensionality_reduction_method == 'UMAP':
        reducer = umap.UMAP(random_state=42)
        n_components = dimensionality_reduction_params.get('n_components', 2)
        X_reduced = reducer.fit_transform(df_selected)
        df_reduced = pd.DataFrame(X_reduced, columns=[f"UMAP_{i+1}" for i in range(n_components)])
    else:
        df_reduced = df_selected  # Если метод не выбран, использовать выбранные признаки

    # Этап Кластеризации
    clustering_algorithm = job.parameters.clustering_algorithm
    clustering_params = job.parameters.parameters.get('clustering', {})

    if clustering_algorithm == 'DBSCAN':
        eps = clustering_params.get('eps', 1.5)
        min_samples = clustering_params.get('min_samples', 5)
        df_clustered = dbscan_clustering(df_reduced, eps=eps, min_samples=min_samples)
    elif clustering_algorithm == 'GMM':
        n_components = clustering_params.get('n_components', 6)
        random_state = clustering_params.get('random_state', 42)
        df_clustered = gmm_clustering(df_reduced, n_components=n_components, random_state=random_state)
    elif clustering_algorithm == 'Hierarchical':
        n_clusters = clustering_params.get('n_clusters', 3)
        linkage = clustering_params.get('linkage', 'ward')
        df_clustered = hierarchical_clustering(df_reduced, n_clusters=n_clusters, linkage=linkage)
    elif clustering_algorithm == 'KMeans':
        n_clusters = clustering_params.get('n_clusters', 3)
        random_state = clustering_params.get('random_state', 42)
        df_clustered = kmeans_clustering(df_reduced, n_clusters=n_clusters, random_state=random_state)
    elif clustering_algorithm == 'OPTICS':
        min_samples = clustering_params.get('min_samples', 10)
        xi = clustering_params.get('xi', 0.05)
        min_cluster_size = clustering_params.get('min_cluster_size', 0.1)
        df_clustered = optics_clustering(df_reduced, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    elif clustering_algorithm == 'Spectral':
        n_clusters = clustering_params.get('n_clusters', 6)
        affinity = clustering_params.get('affinity', 'rbf')
        gamma = clustering_params.get('gamma', 1.0)
        df_clustered = spectral_clustering(df_reduced, n_clusters=n_clusters, affinity=affinity, gamma=gamma)
    else:
        # Если метод не выбран или неизвестен, используем KMeans по умолчанию
        n_clusters = 3
        random_state = 42
        df_clustered = kmeans_clustering(df_reduced, n_clusters=n_clusters, random_state=random_state)

    # Добавление меток кластеров к исходному датасету
    df_final['Cluster'] = df_clustered['Cluster']

    # Вычисление метрик
    metrics = calculate_metrics(df_final, features=selected_features, cluster_column='Cluster')
    job.metrics = metrics

    # Создание визуализаций
    visualizations = visualize_clusters(
        df_final,
        cluster_column='Cluster',
        features=selected_features,
        job_id=str(job.id)
    )
    job.visualizations = visualizations

    # Сохранение результата
    result_csv = df_final.to_csv(index=False)
    job.result_file.save(f"result_{job.id}.csv", ContentFile(result_csv))
    job.status = 'Completed'
    job.completed_at = pd.Timestamp.now()
    job.save()
