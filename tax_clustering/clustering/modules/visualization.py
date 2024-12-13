import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import scipy.cluster.hierarchy as sch
import umap
import os
from django.conf import settings
from django.core.files.base import ContentFile
import base64
import logging

logger = logging.getLogger(__name__)

def visualize_clusters(data, cluster_column, features, job_id):
    """
    Функция для создания визуализаций:
    - PCA
    - t-SNE
    - UMAP
    - Матрица расстояний
    - Дендограмма
    - Scatter Plot
    
    :param data: DataFrame, содержащий данные и метки кластеров
    :param cluster_column: str, название столбца с метками кластеров
    :param features: список с названиями столбцов, содержащих признаки
    :param job_id: str, ID задания кластеризации для сохранения файлов
    :return: Список URL-адресов сгенерированных изображений
    """
    try:
        # Извлечение данных
        X = data[features].values
        y = data[cluster_column].values

        # Уникальные метки кластеров
        clusters = np.unique(y)
        palette = sns.color_palette('hsv', len(clusters))

        # Создание директории для сохранения визуализаций
        visuals_dir = os.path.join(settings.MEDIA_ROOT, 'visualizations', job_id)
        os.makedirs(visuals_dir, exist_ok=True)

        visual_urls = []

        # 1. PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        for i, cluster in enumerate(clusters):
            plt.scatter(
                X_pca[y == cluster, 0],
                X_pca[y == cluster, 1],
                label=f'Cluster {cluster}',
                color=palette[i]
            )
        plt.title('PCA Visualization')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        pca_path = os.path.join(visuals_dir, 'pca.png')
        plt.savefig(pca_path)
        plt.close()
        visual_urls.append(os.path.join(settings.MEDIA_URL, 'visualizations', job_id, 'pca.png'))

        # 2. t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        plt.figure(figsize=(8, 6))
        for i, cluster in enumerate(clusters):
            plt.scatter(
                X_tsne[y == cluster, 0],
                X_tsne[y == cluster, 1],
                label=f'Cluster {cluster}',
                color=palette[i]
            )
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE1')
        plt.ylabel('t-SNE2')
        plt.legend()
        tsne_path = os.path.join(visuals_dir, 'tsne.png')
        plt.savefig(tsne_path)
        plt.close()
        visual_urls.append(os.path.join(settings.MEDIA_URL, 'visualizations', job_id, 'tsne.png'))

        # 3. UMAP
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)
        plt.figure(figsize=(8, 6))
        for i, cluster in enumerate(clusters):
            plt.scatter(
                X_umap[y == cluster, 0],
                X_umap[y == cluster, 1],
                label=f'Cluster {cluster}',
                color=palette[i]
            )
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
        umap_path = os.path.join(visuals_dir, 'umap.png')
        plt.savefig(umap_path)
        plt.close()
        visual_urls.append(os.path.join(settings.MEDIA_URL, 'visualizations', job_id, 'umap.png'))

        # 4. Матрица расстояний
        dist_matrix = pairwise_distances(X)
        plt.figure(figsize=(10, 8))
        sns.heatmap(dist_matrix, cmap='viridis')
        plt.title('Distance Matrix')
        heatmap_path = os.path.join(visuals_dir, 'distance_matrix.png')
        plt.savefig(heatmap_path)
        plt.close()
        visual_urls.append(os.path.join(settings.MEDIA_URL, 'visualizations', job_id, 'distance_matrix.png'))

        # 5. Дендограмма
        linkage_matrix = sch.linkage(X, method='ward')
        plt.figure(figsize=(12, 8))
        sch.dendrogram(linkage_matrix, labels=y)
        plt.title('Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        dendrogram_path = os.path.join(visuals_dir, 'dendrogram.png')
        plt.savefig(dendrogram_path)
        plt.close()
        visual_urls.append(os.path.join(settings.MEDIA_URL, 'visualizations', job_id, 'dendrogram.png'))

        # 6. Scatter Plot
        plt.figure(figsize=(8, 6))
        for i, cluster in enumerate(clusters):
            plt.scatter(
                X[y == cluster, 0],
                X[y == cluster, 1],
                label=f'Cluster {cluster}',
                color=palette[i]
            )
        plt.title('Scatter Plot of Features')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend()
        scatter_path = os.path.join(visuals_dir, 'scatter_plot.png')
        plt.savefig(scatter_path)
        plt.close()
        visual_urls.append(os.path.join(settings.MEDIA_URL, 'visualizations', job_id, 'scatter_plot.png'))

        logger.info(f"Визуализации успешно созданы для задания {job_id}.")
        return visual_urls

    except Exception as e:
        logger.error(f"Ошибка при создании визуализаций для задания {job_id}: {e}", exc_info=True)
        return []
