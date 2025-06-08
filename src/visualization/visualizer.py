import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.config.config import EVALUATION_METRICS

class Visualizer:
    @staticmethod
    def plot_tsne(tsne_results, title="t-SNE Visualization"):
        """Plot t-SNE results."""
        plt.figure(figsize=(16, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_kmeans_clusters(embeddings, clusters, num_clusters):
        """Plot K-means clustering results."""
        plt.figure(figsize=(10, 8))
        cmap = plt.cm.get_cmap('Spectral', num_clusters)
        
        for i in range(num_clusters):
            cluster_indices = np.where(clusters == i)[0]
            plt.scatter(
                embeddings[cluster_indices, 0],
                embeddings[cluster_indices, 1],
                label=f'Cluster {i+1}',
                s=50,
                cmap=cmap,
                color=cmap(i / num_clusters)
            )
        
        plt.title('K-Means Clustering Results')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_dbscan_clusters(embeddings, clusters):
        """Plot DBSCAN clustering results."""
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(clusters)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'k'
            
            class_member_mask = (clusters == k)
            xy = embeddings[class_member_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                'o',
                markerfacecolor=col,
                markeredgecolor='k',
                markersize=6,
                label=f'Cluster {k}' if k != -1 else 'Noise'
            )

        plt.title('DBSCAN Clustering Results')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_elbow_curve(K, inertias):
        """Plot elbow curve for K-means."""
        plt.figure(figsize=(8, 4))
        plt.plot(K, inertias, 'bo-')
        plt.xlabel('Number of clusters, k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(model_name):
        """Plot confusion matrix for model evaluation."""
        metrics = EVALUATION_METRICS[model_name]
        cm = np.array([
            [metrics['true_negative'], metrics['false_positive']],
            [metrics['false_negative'], metrics['true_positive']]
        ])

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Theme 1', 'Theme 2'],
            yticklabels=['Theme 1', 'Theme 2']
        )
        plt.xlabel('Predicted Themes')
        plt.ylabel('True Themes')
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.show()

    @staticmethod
    def plot_metrics_comparison():
        """Plot comparison of metrics between models."""
        results = {
            'Metric': ['Accuracy', 'Precision', 'Recall'],
            'Baseline TF-IDF': [0.38, 0.38, 0.40],
            'XLNet Based Architecture': [0.62, 0.56, 0.64]
        }

        df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', x='Metric', ax=ax)
        
        plt.title('Comparison of Accuracy, Precision, and Recall')
        plt.ylabel('Scores')
        plt.xlabel('Metrics')
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.show() 