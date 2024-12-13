import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)

def dbscan_clustering(df, eps=1.5, min_samples=5):
    """
    Применяет DBSCAN для кластеризации данных.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(df)
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    return df_clustered

def gmm_clustering(df, n_components=6, random_state=42):
    """
    Применяет Gaussian Mixture Model для кластеризации данных.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)
    labels = gmm.predict(df)
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    return df_clustered

def hierarchical_clustering(df, n_clusters=3, linkage='ward'):
    """
    Применяет иерархическую кластеризацию для данных.
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = clustering.fit_predict(df)
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    return df_clustered

def kmeans_clustering(df, n_clusters=3, random_state=42):
    """
    Применяет K-Means для кластеризации данных.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(df)
    labels = kmeans.labels_
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    return df_clustered

def optics_clustering(df, min_samples=10, xi=0.05, min_cluster_size=0.1):
    """
    Применяет OPTICS для кластеризации данных.
    """
    clustering = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    labels = clustering.fit_predict(df)
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    return df_clustered

def spectral_clustering(df, n_clusters=6, affinity='rbf', gamma=1.0):
    """
    Применяет Spectral Clustering для кластеризации данных.
    """
    clustering = SpectralClustering(n_clusters=n_clusters, affinity=affinity, gamma=gamma, random_state=42)
    labels = clustering.fit_predict(df)
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    return df_clustered
