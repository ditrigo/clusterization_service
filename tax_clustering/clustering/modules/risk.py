import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from django.conf import settings


def compute_risk_score(data_row):
    """
    Принимает DataFrame с одной строкой и возвращает скаляр – сумму
    всех числовых полей этой строки.
    """
    numeric = data_row.select_dtypes(include=[np.number]).iloc[0]
    return float(numeric.sum())

def generate_shap_plot(model, data_row: pd.DataFrame, feature_names: list, save_path: str):
    """
    Генерация SHAP графика для объяснения риска с отключенной проверкой аддитивности.
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional",
        model_output="raw"
    )
    shap_values = explainer(data_row, check_additivity=False)

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path
