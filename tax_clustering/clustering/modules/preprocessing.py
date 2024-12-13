import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

def determine_strategy(column, missing_ratio_threshold=0.5):
    """Определяет способ обработки пропусков в зависимости от характеристик данных."""
    missing_ratio = column.isnull().mean()

    # Если пропусков больше заданного порога, столбец лучше удалить
    if missing_ratio > missing_ratio_threshold:
        return 'drop'

    # Числовые данные
    if column.dtype in ['float64', 'int64']:
        # Проверяем наличие выбросов с помощью межквартильного размаха (IQR)
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        has_outliers = ((column < (Q1 - outlier_threshold)) | (column > (Q3 + outlier_threshold))).any()

        if has_outliers:
            return 'median'  # Если есть выбросы, используем медиану
        else:
            return 'mean'  # Иначе используем среднее значение

    # Категориальные данные
    elif column.dtype == 'object':
        return 'mode'  # Для категорий заполняем модой

    return 'zero'  # На всякий случай, если тип неизвестен

def handle_missing_values_auto(data, missing_ratio_threshold=0.5):
    """Автоматически обрабатывает пропуски на основе анализа данных."""
    strategies_applied = {}

    # Анализируем каждый столбец
    for column_name in data.columns:
        column = data[column_name]
        if column.isnull().sum() > 0:  # Если есть пропуски
            strategy = determine_strategy(column, missing_ratio_threshold)
            
            # Применяем стратегию
            if strategy == 'drop':
                data = data.drop(columns=[column_name])
            elif strategy == 'mean':
                data[column_name].fillna(column.mean(), inplace=True)
            elif strategy == 'median':
                data[column_name].fillna(column.median(), inplace=True)
            elif strategy == 'mode':
                data[column_name].fillna(column.mode()[0], inplace=True)
            elif strategy == 'zero':
                data[column_name].fillna(0, inplace=True)
            
            strategies_applied[column_name] = strategy

    return data, strategies_applied

def remove_outliers(df, numeric_columns, outlier_percentage=0.03):
    """Удаляет выбросы из числовых столбцов."""
    df_no_outliers = df.copy()
    outlier_info = {}

    for column in numeric_columns:
        Q1 = df_no_outliers[column].quantile(outlier_percentage)
        Q3 = df_no_outliers[column].quantile(1 - outlier_percentage)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_shape = df_no_outliers.shape
        df_no_outliers = df_no_outliers[(df_no_outliers[column] >= lower_bound) & (df_no_outliers[column] <= upper_bound)]
        final_shape = df_no_outliers.shape
        outlier_info[column] = {'initial_rows': initial_shape[0], 'final_rows': final_shape[0]}

    return df_no_outliers, outlier_info

def analyze_distributions(df, numeric_columns):
    """Анализирует распределение данных и определяет лучшее распределение для каждого столбца."""
    column_distribution = []

    for column in numeric_columns:
        series = df[column]
        best_fit = None
        best_pvalue = 0

        for dist_name in ['norm', 'expon', 'lognorm', 'uniform']:
            dist = getattr(stats, dist_name)
            params = dist.fit(series)
            ks_stat, p_value = stats.kstest(series, dist_name, args=params)
            
            if p_value > best_pvalue:
                best_fit = dist_name
                best_pvalue = p_value

        column_distribution.append(best_fit)

    return column_distribution

def apply_transformations(df, column_distributions):
    """Применяет преобразования к столбцам в зависимости от их распределения."""
    df_transformed = df.copy()
    transform_info = {}

    for i, distribution in enumerate(column_distributions):
        column = df_transformed.columns[i]
        series = df_transformed[column]

        if distribution == 'norm':
            transformed_series = np.sqrt(series.clip(lower=0))
            transform_info[column] = 'sqrt'
        elif distribution == 'expon':
            transformed_series = np.log1p(series.clip(lower=0))
            transform_info[column] = 'log'
        elif distribution == 'lognorm':
            positive_series = series.clip(lower=1e-6)  # Заменяем 0 на очень малое число
            transformed_series, _ = boxcox(positive_series)
            transform_info[column] = 'box-cox'
        elif distribution == 'uniform':
            transformed_series = series
            transform_info[column] = 'none'
        else:
            transformed_series = series
            transform_info[column] = 'unknown'

        df_transformed[column] = transformed_series

    return df_transformed, transform_info

def apply_scalers(df, column_distributions):
    """Применяет масштабаторы к столбцам в зависимости от их распределения."""
    df_scaled = df.copy()
    scaler_info = {}
    columns_to_drop = []

    for i, distribution in enumerate(column_distributions):
        column = df_scaled.columns[i]
        series = df_scaled[column].values.reshape(-1, 1)  # Преобразуем в 2D для масштабаторов

        if distribution == 'norm':
            scaler = StandardScaler()
            scaled_series = scaler.fit_transform(series)
            scaler_info[column] = 'StandardScaler'
        elif distribution in ['expon', 'lognorm']:
            scaler = RobustScaler()
            scaled_series = scaler.fit_transform(series)
            scaler_info[column] = 'RobustScaler'
        elif distribution == 'uniform':
            scaler = MinMaxScaler()
            scaled_series = scaler.fit_transform(series)
            scaler_info[column] = 'MinMaxScaler'
        else:
            # Удаляем столбец, если распределение неизвестно
            columns_to_drop.append(column)
            scaler_info[column] = 'Dropped'
            continue

        # Обновляем столбец в DataFrame
        df_scaled[column] = scaled_series.flatten()

    # Удаляем столбцы, отмеченные для удаления
    df_scaled.drop(columns=columns_to_drop, inplace=True)

    return df_scaled, scaler_info, df_scaled  # Возвращаем df_scaled вместо df_copy
