import pandas as pd
import os
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
from django.core.files.base import ContentFile
from django.conf import settings
import logging
import umap.umap_ as umap


logger = logging.getLogger(__name__)

def execute_preprocessing(job):
    try:
        # Загрузка датасета
        dataset_path = job.dataset.file.path
        df = pd.read_csv(dataset_path)

        # Предобработка
        df_cleaned, _ = handle_missing_values_auto(df, missing_ratio_threshold=0.4)
        numeric_columns = df_cleaned.select_dtypes(include='number').columns.tolist()
        df_no_outliers, _ = remove_outliers(df_cleaned, numeric_columns=numeric_columns)
        column_distributions = analyze_distributions(df_no_outliers, numeric_columns=numeric_columns)
        df_transformed, _ = apply_transformations(df_no_outliers, column_distributions)
        df_scaled, _, df_final = apply_scalers(df_transformed, column_distributions)

        # Сохранение промежуточных данных
        intermediate_dir = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id))
        os.makedirs(intermediate_dir, exist_ok=True)
        preprocessed_path = os.path.join(intermediate_dir, 'preprocessed.csv')
        df_final.to_csv(preprocessed_path, index=False)

        # Обновление статуса задания
        job.preprocessing_completed = True
        job.save()

        logger.info(f"Preprocessing completed for job {job.id}")
    except Exception as e:
        logger.error(f"Error during preprocessing for job {job.id}: {e}", exc_info=True)
        raise e

def execute_feature_selection(job):
    try:
        if not job.preprocessing_completed:
            raise Exception("Preprocessing not completed.")

        # Загрузка промежуточных данных
        intermediate_dir = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id))
        preprocessed_path = os.path.join(intermediate_dir, 'preprocessed.csv')
        df_final = pd.read_csv(preprocessed_path)

        # Отбор признаков
        feature_selection_method = job.parameters.feature_selection_method
        feature_selection_params = job.parameters.parameters.get('feature_selection', {})

        if feature_selection_method == 'Correlation':
            threshold = feature_selection_params.get('threshold', 0.6)
            df_selected, selected_features = correlation_based_selection(df_final, threshold=threshold)
        elif feature_selection_method == 'KMeans':
            n_clusters = feature_selection_params.get('n_clusters', 3)
            top_n = feature_selection_params.get('top_n', 5)
            df_selected, selected_features = kmeans_based_selection(df_final, n_clusters=n_clusters, top_n=top_n)
        elif feature_selection_method == 'Mutual Information':
            n_clusters = feature_selection_params.get('n_clusters', 3)
            top_n = feature_selection_params.get('top_n', 5)
            df_selected, selected_features = mutual_info_selection(df_final, n_clusters=n_clusters, top_n=top_n)
        elif feature_selection_method == 'PCA':
            n_components = feature_selection_params.get('n_components', 5)
            df_selected, _ = pca_selection(df_final, n_components=n_components)
            selected_features = df_selected.columns.tolist()
        elif feature_selection_method == 't-SNE':
            n_components = feature_selection_params.get('n_components', 2)
            perplexity = feature_selection_params.get('perplexity', 30)
            learning_rate = feature_selection_params.get('learning_rate', 200)
            df_selected, _ = t_sne_selection(
                df_final,
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                random_state=42,
                method='exact'
            )
            selected_features = df_selected.columns.tolist()
        elif feature_selection_method == 'Variance Threshold':
            threshold = feature_selection_params.get('threshold', 0.1)
            df_selected, selected_features = variance_threshold_selection(df_final, threshold=threshold)
        elif feature_selection_method == 'Autoencoder':
            encoding_dim = feature_selection_params.get('encoding_dim', 10)
            epochs = feature_selection_params.get('epochs', 50)
            batch_size = feature_selection_params.get('batch_size', 32)
            df_selected = autoencoder_selection(df_final, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
            selected_features = df_selected.columns.tolist()
        elif feature_selection_method == 'Kernel PCA':
            n_components = feature_selection_params.get('n_components', 10)
            kernel = feature_selection_params.get('kernel', 'rbf')
            gamma = feature_selection_params.get('gamma', None)
            df_selected = kernel_pca_selection(df_final, n_components=n_components, kernel=kernel, gamma=gamma)
            selected_features = df_selected.columns.tolist()
        elif feature_selection_method == 'Factor Analysis':
            n_components = feature_selection_params.get('n_components', 10)
            df_selected = factor_analysis_selection(df_final, n_components=n_components)
            selected_features = df_selected.columns.tolist()
        elif feature_selection_method == 'UMAP':
            n_neighbors = feature_selection_params.get('n_neighbors', 15)
            min_dist = feature_selection_params.get('min_dist', 0.1)
            n_components = feature_selection_params.get('n_components', 10)
            random_state = feature_selection_params.get('random_state', 42)
            df_selected = umap_selection(
                df_final,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_state
            )
            selected_features = df_selected.columns.tolist()
        else:
            # Если метод не выбран или неизвестен, используем исходные данные
            df_selected = df_final
            selected_features = df_final.columns.tolist()

        # Сохранение промежуточных данных
        feature_selection_path = os.path.join(intermediate_dir, 'feature_selected.csv')
        df_selected.to_csv(feature_selection_path, index=False)

        # Сохранение выбранных признаков в параметрах
        job.parameters.parameters['feature_selection']['selected_features'] = selected_features
        job.parameters.save()

        # Обновление статуса задания
        job.feature_selection_completed = True
        job.save()

        logger.info(f"Feature Selection completed for job {job.id}")
    except Exception as e:
        logger.error(f"Error during feature selection for job {job.id}: {e}", exc_info=True)
        raise e

def execute_dimensionality_reduction(job):
    try:
        if not job.feature_selection_completed:
            raise Exception("Feature selection not completed.")

        # Загрузка промежуточных данных
        intermediate_dir = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id))
        feature_selection_path = os.path.join(intermediate_dir, 'feature_selected.csv')
        df_selected = pd.read_csv(feature_selection_path)

        # Снижение размерности
        dimensionality_reduction_method = job.parameters.dimensionality_reduction_method
        dimensionality_reduction_params = job.parameters.parameters.get('dimensionality_reduction', {})

        if dimensionality_reduction_method == 'UMAP':
            n_components = dimensionality_reduction_params.get('n_components', 2)
            random_state = dimensionality_reduction_params.get('random_state', 42)
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            X_reduced = reducer.fit_transform(df_selected)
            df_reduced = pd.DataFrame(X_reduced, columns=[f"UMAP_{i+1}" for i in range(n_components)])
        else:
            df_reduced = df_selected

        # Сохранение промежуточных данных
        dimensionality_reduction_path = os.path.join(intermediate_dir, 'dimensionality_reduction.csv')
        df_reduced.to_csv(dimensionality_reduction_path, index=False)

        # Обновление статуса задания
        job.dimensionality_reduction_completed = True
        job.save()

        logger.info(f"Dimensionality Reduction completed for job {job.id}")
    except Exception as e:
        logger.error(f"Error during dimensionality reduction for job {job.id}: {e}", exc_info=True)
        raise e

def execute_clustering(job):
    try:
        if not job.dimensionality_reduction_completed:
            raise Exception("Dimensionality reduction not completed.")

        # Загрузка промежуточных данных
        intermediate_dir = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id))
        dimensionality_reduction_path = os.path.join(intermediate_dir, 'dimensionality_reduction.csv')
        df_reduced = pd.read_csv(dimensionality_reduction_path)

        # Кластеризация
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
            # По умолчанию используем KMeans
            n_clusters = 3
            random_state = 42
            df_clustered = kmeans_clustering(df_reduced, n_clusters=n_clusters, random_state=random_state)

        # Загрузка финальных данных
        feature_selection_path = os.path.join(intermediate_dir, 'feature_selected.csv')
        df_final = pd.read_csv(feature_selection_path)
        df_final['Cluster'] = df_clustered['Cluster']

        # Сохранение промежуточных данных
        clustering_path = os.path.join(intermediate_dir, 'clustering.csv')
        df_final.to_csv(clustering_path, index=False)

        # Обновление статуса задания
        job.clustering_completed = True
        job.save()

        logger.info(f"Clustering completed for job {job.id}")
    except Exception as e:
        logger.error(f"Error during clustering for job {job.id}: {e}", exc_info=True)
        raise e

def execute_metrics(job):
    try:
        if not job.clustering_completed:
            raise Exception("Clustering not completed.")

        # Загрузка финальных данных
        intermediate_dir = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id))
        clustering_path = os.path.join(intermediate_dir, 'clustering.csv')
        df_final = pd.read_csv(clustering_path)

        # Получение признаков для метрик
        selected_features = job.parameters.parameters.get('feature_selection', {}).get('selected_features', [])
        if not selected_features:
            selected_features = [col for col in df_final.columns if col != 'Cluster']

        # Вычисление метрик
        metrics = calculate_metrics(df_final, features=selected_features, cluster_column='Cluster')
        job.metrics = metrics
        job.metrics_completed = True
        job.save()

        logger.info(f"Metrics calculated for job {job.id}")
    except Exception as e:
        logger.error(f"Error calculating metrics for job {job.id}: {e}", exc_info=True)
        raise e

def execute_visualization(job):
    try:
        if not job.metrics_completed:
            raise Exception("Metrics calculation not completed.")

        # Загрузка финальных данных
        intermediate_dir = os.path.join(settings.MEDIA_ROOT, 'intermediate', str(job.id))
        clustering_path = os.path.join(intermediate_dir, 'clustering.csv')
        df_final = pd.read_csv(clustering_path)

        # Получение признаков для визуализации
        selected_features = job.parameters.parameters.get('feature_selection', {}).get('selected_features', [])
        if not selected_features:
            selected_features = [col for col in df_final.columns if col != 'Cluster']

        # Создание визуализаций
        visualizations = visualize_clusters(
            data=df_final,
            cluster_column='Cluster',
            features=selected_features,
            job_id=str(job.id)
        )
        job.visualizations = visualizations
        job.visualization_completed = True
        job.save()

        logger.info(f"Visualizations created for job {job.id}")
    except Exception as e:
        logger.error(f"Error creating visualizations for job {job.id}: {e}", exc_info=True)
        raise e

def execute_all_steps(job):
    try:
        # Выполнение всех стадий последовательно
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
        job.completed_at = pd.Timestamp.now()
        job.save()

        logger.info(f"All steps completed for job {job.id}")
    except Exception as e:
        job.status = 'Failed'
        job.save()
        logger.error(f"Error during processing all steps for job {job.id}: {e}", exc_info=True)
        raise e
