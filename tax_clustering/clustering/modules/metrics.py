import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    calinski_harabasz_score
)
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(df, features, cluster_column='Cluster'):
    """
    Функция для вычисления метрик кластеризации:
    - Inertia (инерция)
    - Silhouette Score
    - Davies-Bouldin Score
    - Dunn Index
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Calinski-Harabasz Index
    
    :param df: DataFrame, содержащий данные и столбец с метками кластеров
    :param features: список с названиями столбцов, содержащих признаки
    :param cluster_column: str, название столбца с метками кластеров
    :return: Словарь с метками и значениями метрик
    """
    try:
        # Получение данных и меток кластеров
        X = df[features].values
        y = df[cluster_column].values

        # Проверка наличия как минимум двух кластеров
        unique_clusters = set(y)
        if len(unique_clusters) < 2:
            logger.warning("Недостаточно кластеров для вычисления метрик.")
            return {}

        # 1. Inertia (сумма квадратов расстояний до центров кластеров)
        inertia = np.sum([
            np.sum(cdist(X[y == i], [X[y == i].mean(axis=0)], 'sqeuclidean'))
            for i in unique_clusters if i != -1  # Исключаем шум (если применимо)
        ])

        # 2. Silhouette Score
        silhouette = silhouette_score(X, y)

        # 3. Davies-Bouldin Score
        db_score = davies_bouldin_score(X, y)

        # 4. Dunn Index
        def dunn_index(X, labels):
            clusters = np.unique(labels)
            intra_cluster_distances = []
            inter_cluster_distances = []

            for i in clusters:
                points_in_cluster = X[labels == i]
                if len(points_in_cluster) == 0:
                    continue
                # Внутрикластерное расстояние (максимальное расстояние внутри кластера)
                intra_cluster_distances.append(np.max(cdist(points_in_cluster, points_in_cluster)))

            for i in clusters:
                for j in clusters:
                    if i < j:
                        points_i = X[labels == i]
                        points_j = X[labels == j]
                        if len(points_i) == 0 or len(points_j) == 0:
                            continue
                        # Межклассовое расстояние (минимальное расстояние между кластерами)
                        inter_cluster_distances.append(np.min(cdist(points_i, points_j)))

            if len(inter_cluster_distances) == 0 or len(intra_cluster_distances) == 0:
                return 0

            return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)

        dunn = dunn_index(X, y)

        # 5. Adjusted Rand Index (ARI)
        ari = adjusted_rand_score(y, y)  # Используем метки самих кластеров для оценки

        # 6. Normalized Mutual Information (NMI)
        nmi = normalized_mutual_info_score(y, y)

        # 7. Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(X, y)

        # Сохранение результатов в словарь
        metrics = {
            'Inertia': inertia,
            'Silhouette Score': silhouette,
            'Davies-Bouldin Score': db_score,
            'Dunn Index': dunn,
            'Adjusted Rand Index (ARI)': ari,
            'Normalized Mutual Information (NMI)': nmi,
            'Calinski-Harabasz Index': ch_index
        }

        logger.info("Метрики кластеризации успешно вычислены.")
        return metrics

    except Exception as e:
        logger.error(f"Ошибка при вычислении метрик кластеризации: {e}", exc_info=True)
        return {}
