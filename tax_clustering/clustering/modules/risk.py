# clustering/modules/risk.py
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from django.conf import settings
import numpy as np


def compute_risk_score(data_row):
    """
    Принимает DataFrame с одной строкой и возвращает скаляр – нормированную сумму
    всех числовых полей этой строки.
    """
    numeric = data_row.select_dtypes(include=[np.number]).iloc[0]
    return float(numeric.sum())


def generate_shap_plot(model, data_row, feature_names, save_path):
    """
    Генерация SHAP графика для объяснения риска.
    
    :param model: обученная модель, для которой рассчитаны SHAP значения
    :param data_row: DataFrame с одной записью для которой нужно получить объяснения
    :param feature_names: список имен признаков
    :param save_path: путь, куда сохранить изображение
    :return: путь до сохраненного изображения
    """
    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(data_row)
    
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path
