# frontend/views.py

import requests
from django.shortcuts import render, redirect
from django.urls import reverse
from django.conf import settings
from django.contrib import messages
import uuid

# Базовый URL API
API_BASE_URL = "http://localhost:8000/api"

def home(request):
    return render(request, 'frontend/home.html')

def dataset_list(request):
    response = requests.get(f"{API_BASE_URL}/datasets/")
    if response.status_code == 200:
        datasets = response.json()
    else:
        datasets = []
        messages.error(request, "Не удалось загрузить список датасетов.")
    
    return render(request, 'frontend/dataset_list.html', {'datasets': datasets})


def dataset_detail(request, dataset_id):
    """
    Отображает детали выбранного датасета, включая заголовки и превью (первые 5 строк).
    
    :param request: HTTP-запрос
    :param dataset_id: UUID идентификатор датасета
    :return: HTML страница с деталями датасета
    """
    response = requests.get(f"{API_BASE_URL}/datasets/{dataset_id}/")
    if response.status_code == 200:
        dataset = response.json()
    else:
        dataset = None
        messages.error(request, "Не удалось загрузить детали датасета.")
    
    # Передаём поля columns и preview из ответа API (если они есть)
    context = {
        'dataset': dataset,
        'columns': dataset.get('columns', []) if dataset else [],
        'preview': dataset.get('preview', []) if dataset else []
    }
    return render(request, 'frontend/dataset_detail.html', context)

def upload_dataset(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        file = request.FILES.get('file')
        if not name or not file:
            messages.error(request, "Пожалуйста, заполните все поля.")
            return redirect(reverse('frontend:upload_dataset'))
        
        data = {
            'name': name,
        }
        files = {
            'file': file
        }
        try:
            response = requests.post(f"{API_BASE_URL}/datasets/", data=data, files=files)
            if response.status_code == 201:
                messages.success(request, "Датасет успешно загружен.")
                return redirect(reverse('frontend:dataset_list'))
            else:
                error_detail = response.json()
                messages.error(request, f"Ошибка при загрузке датасета: {error_detail}")
                return redirect(reverse('frontend:upload_dataset'))
        except requests.exceptions.RequestException as e:
            messages.error(request, f"Ошибка подключения к серверу: {e}")
            return redirect(reverse('frontend:upload_dataset'))
    return render(request, 'frontend/upload_dataset.html')

def job_list(request):
    response = requests.get(f"{API_BASE_URL}/clustering-jobs/")
    if response.status_code == 200:
        jobs = response.json()
    else:
        jobs = []
        messages.error(request, "Не удалось загрузить список заданий.")
    
    return render(request, 'frontend/job_list.html', {'jobs': jobs})

def job_detail(request, job_id):
    # Получение деталей задания
    response = requests.get(f"{API_BASE_URL}/clustering-jobs/{job_id}/")
    if response.status_code == 200:
        job = response.json()
    else:
        job = None
        messages.error(request, "Не удалось загрузить детали задания.")
        return redirect(reverse('frontend:job_list'))
    
    # Обработка действий: выполнение стадий или всех шагов
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'execute_step':
            step = request.POST.get('step')
            execute_url = f"{API_BASE_URL}/clustering-jobs/{job_id}/{step}/"
            execute_response = requests.post(execute_url)
            if execute_response.status_code == 200:
                messages.success(request, f"Стадия '{step}' успешно выполнена.")
            else:
                messages.error(request, f"Ошибка при выполнении стадии '{step}'.")
        elif action == 'execute_all_steps':
            execute_url = f"{API_BASE_URL}/clustering-jobs/{job_id}/execute_all_steps/"
            execute_response = requests.post(execute_url)
            if execute_response.status_code == 200:
                messages.success(request, "Все стадии успешно выполнены.")
            else:
                messages.error(request, "Ошибка при выполнении всех стадий.")
        return redirect(reverse('frontend:job_detail', args=[job_id]))
    
    return render(request, 'frontend/job_detail.html', {'job': job})


def create_job(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset')
        feature_selection_method = request.POST.get('feature_selection_method')
        dimensionality_reduction_method = request.POST.get('dimensionality_reduction_method')
        clustering_algorithm = request.POST.get('clustering_algorithm')
        
        # Сбор параметров для отбора признаков
        feature_selection_params = {}
        if feature_selection_method == 'Correlation':
            threshold = float(request.POST.get('threshold', 0.6))
            feature_selection_params['threshold'] = threshold
        elif feature_selection_method == 'KMeans':
            n_clusters = int(request.POST.get('kmeans_n_clusters', 3))
            top_n = int(request.POST.get('kmeans_top_n', 5))
            feature_selection_params['n_clusters'] = n_clusters
            feature_selection_params['top_n'] = top_n
        elif feature_selection_method == 'Mutual Information':
            n_clusters = int(request.POST.get('mi_n_clusters', 3))
            top_n = int(request.POST.get('mi_top_n', 5))
            feature_selection_params['n_clusters'] = n_clusters
            feature_selection_params['top_n'] = top_n
        elif feature_selection_method == 'Variance Threshold':
            threshold = float(request.POST.get('variance_threshold', 0.1))
            feature_selection_params['threshold'] = threshold
        elif feature_selection_method == 'PCA':
            n_components = int(request.POST.get('pca_n_components', 5))
            feature_selection_params['n_components'] = n_components
        elif feature_selection_method == 't-SNE':
            n_components = int(request.POST.get('tsne_n_components', 2))
            perplexity = float(request.POST.get('tsne_perplexity', 30))
            learning_rate = float(request.POST.get('tsne_learning_rate', 200))
            feature_selection_params['n_components'] = n_components
            feature_selection_params['perplexity'] = perplexity
            feature_selection_params['learning_rate'] = learning_rate
        # Добавьте другие методы и их параметры при необходимости

        # Сбор параметров для снижения размерности
        dimensionality_reduction_params = {}
        if dimensionality_reduction_method == 'Autoencoder':
            encoding_dim = int(request.POST.get('autoencoder_encoding_dim', 10))
            epochs = int(request.POST.get('autoencoder_epochs', 50))
            batch_size = int(request.POST.get('autoencoder_batch_size', 32))
            dimensionality_reduction_params['encoding_dim'] = encoding_dim
            dimensionality_reduction_params['epochs'] = epochs
            dimensionality_reduction_params['batch_size'] = batch_size
        elif dimensionality_reduction_method == 'Kernel PCA':
            n_components = int(request.POST.get('kernelpca_n_components', 10))
            kernel = request.POST.get('kernelpca_kernel', 'rbf')
            gamma = request.POST.get('kernelpca_gamma', None)
            if gamma:
                gamma = float(gamma)
            dimensionality_reduction_params['n_components'] = n_components
            dimensionality_reduction_params['kernel'] = kernel
            dimensionality_reduction_params['gamma'] = gamma
        elif dimensionality_reduction_method == 'Factor Analysis':
            n_components = int(request.POST.get('factor_analysis_n_components', 10))
            dimensionality_reduction_params['n_components'] = n_components
        elif dimensionality_reduction_method == 'UMAP':
            n_neighbors = int(request.POST.get('umap_n_neighbors', 15))
            min_dist = float(request.POST.get('umap_min_dist', 0.1))
            n_components = int(request.POST.get('umap_n_components', 10))
            random_state = int(request.POST.get('umap_random_state', 42))
            dimensionality_reduction_params['n_neighbors'] = n_neighbors
            dimensionality_reduction_params['min_dist'] = min_dist
            dimensionality_reduction_params['n_components'] = n_components
            dimensionality_reduction_params['random_state'] = random_state
        # Добавьте другие методы и их параметры при необходимости

        # Сбор параметров для кластеризации
        clustering_params = {}
        if clustering_algorithm == 'DBSCAN':
            eps = float(request.POST.get('dbscan_eps', 1.5))
            min_samples = int(request.POST.get('dbscan_min_samples', 5))
            clustering_params['eps'] = eps
            clustering_params['min_samples'] = min_samples
        elif clustering_algorithm == 'GMM':
            n_components = int(request.POST.get('gmm_n_components', 6))
            random_state = int(request.POST.get('gmm_random_state', 42))
            clustering_params['n_components'] = n_components
            clustering_params['random_state'] = random_state
        elif clustering_algorithm == 'Hierarchical':
            n_clusters = int(request.POST.get('hierarchical_n_clusters', 3))
            linkage = request.POST.get('hierarchical_linkage', 'ward')
            clustering_params['n_clusters'] = n_clusters
            clustering_params['linkage'] = linkage
        elif clustering_algorithm == 'KMeans':
            n_clusters = int(request.POST.get('kmeans_n_clusters', 3))
            random_state = int(request.POST.get('kmeans_random_state', 42))
            clustering_params['n_clusters'] = n_clusters
            clustering_params['random_state'] = random_state
        elif clustering_algorithm == 'OPTICS':
            min_samples = int(request.POST.get('optics_min_samples', 10))
            xi = float(request.POST.get('optics_xi', 0.05))
            min_cluster_size = float(request.POST.get('optics_min_cluster_size', 0.1))
            clustering_params['min_samples'] = min_samples
            clustering_params['xi'] = xi
            clustering_params['min_cluster_size'] = min_cluster_size
        elif clustering_algorithm == 'Spectral':
            n_clusters = int(request.POST.get('spectral_n_clusters', 6))
            affinity = request.POST.get('spectral_affinity', 'rbf')
            gamma = float(request.POST.get('spectral_gamma', 1.0))
            clustering_params['n_clusters'] = n_clusters
            clustering_params['affinity'] = affinity
            clustering_params['gamma'] = gamma
        # Добавьте другие методы и их параметры при необходимости

        payload = {
            "dataset": dataset_id,
            "parameters": {
                "feature_selection_method": feature_selection_method,
                "dimensionality_reduction_method": dimensionality_reduction_method,
                "clustering_algorithm": clustering_algorithm,
                "feature_selection": feature_selection_params,
                "dimensionality_reduction": dimensionality_reduction_params,
                "clustering": clustering_params
            }
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{API_BASE_URL}/clustering-jobs/", json=payload, headers=headers)
        if response.status_code == 201:
            messages.success(request, "Кластеризационное задание успешно создано.")
            return redirect(reverse('frontend:job_list'))
        else:
            # Получаем подробные ошибки из ответа API
            try:
                error_detail = response.json()
            except ValueError:
                error_detail = response.text
            messages.error(request, f"Ошибка при создании задания: {error_detail}")
            return redirect(reverse('frontend:create_job'))
    
    # Получение списка датасетов для выбора в форме
    response = requests.get(f"{API_BASE_URL}/datasets/")
    if response.status_code == 200:
        datasets = response.json()
    else:
        datasets = []
        messages.error(request, "Не удалось загрузить список датасетов.")
    
    return render(request, 'frontend/create_job.html', {'datasets': datasets})

def execute_all_steps(request, job_id):
    if request.method == 'POST':
        response = requests.post(f"{API_BASE_URL}/clustering-jobs/{job_id}/execute_all_steps/")
        if response.status_code == 200:
            messages.success(request, "Все этапы выполнены успешно.")
        else:
            error_detail = response.json().get('detail', 'Неизвестная ошибка.')
            messages.error(request, f"Ошибка при выполнении этапов: {error_detail}")
        return redirect(reverse('frontend:job_detail', args=[job_id]))
    else:
        return redirect(reverse('frontend:job_detail', args=[job_id]))