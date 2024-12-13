import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, FactorAnalysis
import umap.umap_ as umap
import matplotlib.pyplot as plt

def autoencoder_selection(df, encoding_dim=10, epochs=50, batch_size=32):
    """
    Применяет Autoencoder для создания латентного пространства.
    
    :param df: pandas DataFrame, исходные данные
    :param encoding_dim: int, размерность латентного пространства
    :param epochs: int, количество эпох обучения
    :param batch_size: int, размер батча
    :return: pandas DataFrame с латентными признаками
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    input_dim = X_scaled.shape[1]
    
    # Определение архитектуры Autoencoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    encoder = models.Model(inputs=input_layer, outputs=encoded)
    
    # Компиляция и обучение модели
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    
    # Получение латентного пространства
    df_autoencoded = encoder.predict(X_scaled)
    df_latent = pd.DataFrame(df_autoencoded, columns=[f"autoenc_{i+1}" for i in range(encoding_dim)])
    
    return df_latent

def kernel_pca_selection(df, n_components=10, kernel='rbf', gamma=None):
    """
    Применяет Kernel PCA для создания латентного пространства.
    
    :param df: pandas DataFrame, исходные данные
    :param n_components: int, количество компонентов
    :param kernel: str, тип ядра
    :param gamma: float, параметр для некоторых ядер
    :return: pandas DataFrame с латентными признаками
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, fit_inverse_transform=False)
    X_kpca = kpca.fit_transform(X_scaled)
    
    df_kpca = pd.DataFrame(X_kpca, columns=[f"kernelpca_{i+1}" for i in range(n_components)])
    return df_kpca

def factor_analysis_selection(df, n_components=10):
    """
    Применяет Factor Analysis для создания латентного пространства.
    
    :param df: pandas DataFrame, исходные данные
    :param n_components: int, количество факторов
    :return: pandas DataFrame с латентными признаками
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    fa = FactorAnalysis(n_components=n_components, random_state=42)
    X_fa = fa.fit_transform(X_scaled)
    
    df_fa = pd.DataFrame(X_fa, columns=[f"factor_{i+1}" for i in range(n_components)])
    return df_fa

def umap_selection(df, n_neighbors=15, min_dist=0.1, n_components=10, random_state=42):
    """
    Применяет UMAP для создания латентного пространства.
    
    :param df: pandas DataFrame, исходные данные
    :param n_neighbors: int, количество соседей
    :param min_dist: float, минимальное расстояние
    :param n_components: int, количество компонент
    :param random_state: int, для воспроизводимости
    :return: pandas DataFrame с латентными признаками
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    X_umap = umap_model.fit_transform(X_scaled)
    
    df_umap = pd.DataFrame(X_umap, columns=[f"umap_{i+1}" for i in range(n_components)])
    return df_umap
