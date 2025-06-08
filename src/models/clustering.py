import umap
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from src.config.config import UMAP_PARAMS, KMEANS_PARAMS

class ClusteringModel:
    def __init__(self):
        self.umap_reducer = umap.UMAP(**UMAP_PARAMS)
        self.kmeans = KMeans(**KMEANS_PARAMS)
        self.dbscan = DBSCAN(eps=0.5, min_samples=10)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)

    def apply_umap(self, embeddings):
        """Apply UMAP dimensionality reduction."""
        return self.umap_reducer.fit_transform(embeddings)

    def apply_kmeans(self, embeddings):
        """Apply K-means clustering."""
        return self.kmeans.fit_predict(embeddings)

    def apply_dbscan(self, embeddings):
        """Apply DBSCAN clustering."""
        return self.dbscan.fit_predict(embeddings)

    def apply_hdbscan(self, embeddings):
        """Apply HDBSCAN clustering."""
        return self.hdbscan.fit_predict(embeddings)

    def get_optimal_clusters(self, embeddings, max_k=10):
        """Find optimal number of clusters using elbow method."""
        inertias = []
        K = range(1, max_k + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
            inertias.append(kmeans.inertia_)
            
        return K, inertias 