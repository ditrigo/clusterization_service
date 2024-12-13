import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import boxcox

def correlation_based_selection(df, threshold=0.6):
    """
    Удаляет признаки с высокой корреляцией.

    :param df: pandas DataFrame, исходные данные
    :param threshold: float, порог корреляции для удаления признаков
    :return: pandas DataFrame, уменьшенный набор данных
    """
    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop

def kmeans_based_selection(df, n_clusters=3, top_n=5):
    """
    Выбирает топ-N признаков с наибольшей дисперсией кластеров.

    :param df: pandas DataFrame, исходные данные
    :param n_clusters: int, количество кластеров для KMeans
    :param top_n: int, количество признаков для выбора
    :return: pandas DataFrame, уменьшенный набор данных
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df)
    cluster_variances = np.var(kmeans.cluster_centers_, axis=0)
    selected_features = df.columns[np.argsort(cluster_variances)[-top_n:]]
    df_selected = df[selected_features]
    return df_selected, selected_features.tolist()

def mutual_info_selection(df, n_clusters=3, top_n=5):
    """
    Выбирает топ-N признаков на основе взаимной информации с метками кластеров.

    :param df: pandas DataFrame, исходные данные
    :param n_clusters: int, количество кластеров для KMeans
    :param top_n: int, количество признаков для выбора
    :return: pandas DataFrame, уменьшенный набор данных
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    mi_scores = mutual_info_classif(df, cluster_labels)
    selected_features = df.columns[mi_scores.argsort()[-top_n:]]
    df_selected = df[selected_features]
    return df_selected, selected_features.tolist()

def pca_selection(df, n_components=5):
    """
    Применяет PCA для снижения размерности и выбирает главные компоненты.

    :param df: pandas DataFrame, исходные данные
    :param n_components: int, количество главных компонент
    :return: pandas DataFrame, DataFrame с главными компонентами
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=[f"PCA_{i+1}" for i in range(n_components)])
    return X_pca_df, pca.explained_variance_ratio_.tolist()

def t_sne_selection(df, n_components=2, perplexity=30, learning_rate=200, random_state=42, method='exact'):
    """
    Применяет t-SNE для снижения размерности.

    :param df: pandas DataFrame, исходные данные
    :param n_components: int, количество компонент
    :param perplexity: float, параметр perplexity
    :param learning_rate: float, скорость обучения
    :param random_state: int, для воспроизводимости
    :param method: str, метод t-SNE ('barnes_hut' или 'exact')
    :return: pandas DataFrame, DataFrame с компонентами t-SNE
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
        method=method
    )
    X_tsne = tsne.fit_transform(X_scaled)
    X_tsne_df = pd.DataFrame(X_tsne, columns=[f"t-SNE_{i+1}" for i in range(n_components)])
    return X_tsne_df, tsne

def variance_threshold_selection(df, threshold=0.1):
    """
    Удаляет признаки с дисперсией ниже порога.

    :param df: pandas DataFrame, исходные данные
    :param threshold: float, порог дисперсии
    :return: pandas DataFrame, уменьшенный набор данных
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    selected_features = df.columns[selector.get_support()]
    df_selected = df[selected_features]
    return df_selected, selected_features.tolist()
