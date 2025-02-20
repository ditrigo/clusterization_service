PRESETS = [
    {
        "name": "Числовые данные",
        "feature_selection": {"method": "Correlation", "threshold": 0.7},
        "dimensionality_reduction": {"method": "Kernel PCA", "n_components": 5, "gamma": 2},
        "clustering": {"algorithm": "KMeans", "n_clusters": 3, "random_state": 42}
    },
    {
        "name": "Категориальные данные",
        "feature_selection": {"method": "Variance Threshold"},
        "dimensionality_reduction": {"method": "t-SNE", "n_components": 2, "perplexity": 30, "learning_rate": 200},
        "clustering": {"algorithm": "Hierarchical", "n_clusters": 4, "linkage": "complete"}
    },
    {
        "name": "Временные ряды",
        "feature_selection": {"method": "Correlation", "threshold": 0.6},
        "dimensionality_reduction": {"method": "Autoencoder", "encoding_dim": 10, "epochs": 50, "batch_size": 32},
        "clustering": {"algorithm": "DBSCAN", "eps": 1.0, "min_samples": 5}
    },
    {
        "name": "Графические координаты",
        "feature_selection": {"method": "Correlation", "threshold": 0.8},
        "dimensionality_reduction": {"method": "UMAP", "n_neighbors": 10, "min_dist": 0.05, "n_components": 2, "random_state": 42},
        "clustering": {"algorithm": "DBSCAN", "eps": 0.1, "min_samples": 3}
    },
    {
        "name": "Данные с шумом",
        "feature_selection": {"method": "Variance Threshold", "threshold": 0.05},
        "dimensionality_reduction": {"method": "UMAP", "n_neighbors": 15, "min_dist": 0.2, "n_components": 3, "random_state": 42},
        "clustering": {"algorithm": "OPTICS", "min_samples": 10, "xi": 0.05, "min_cluster_size": 0.2}
    }
]