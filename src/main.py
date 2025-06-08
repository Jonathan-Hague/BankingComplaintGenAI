import os
from src.models.embedding_model import EmbeddingModel
from src.models.clustering import ClusteringModel
from src.models.theme_extractor import ThemeExtractor
from src.data.data_loader import DataLoader
from src.visualization.visualizer import Visualizer
from src.config.config import (
    COMPLAINTS_FILE,
    EMBEDDINGS_FILE,
    UMAP_CLUSTERS_FILE,
    NUM_CLUSTERS
)

def main():
    # Initialize models
    embedding_model = EmbeddingModel()
    clustering_model = ClusteringModel()
    theme_extractor = ThemeExtractor()
    data_loader = DataLoader(embedding_model.get_device())
    visualizer = Visualizer()

    # Step 1: Load and process complaints
    print("Loading complaints...")
    sentences = data_loader.load_complaints(COMPLAINTS_FILE)

    # Step 2: Generate embeddings
    print("Generating embeddings...")
    embeddings = data_loader.get_embeddings(sentences, embedding_model.get_model())
    data_loader.save_embeddings(embeddings, EMBEDDINGS_FILE)

    # Step 3: Apply UMAP
    print("Applying UMAP...")
    umap_embeddings = clustering_model.apply_umap(embeddings)
    visualizer.plot_tsne(umap_embeddings, "UMAP Visualization")

    # Step 4: Find optimal number of clusters
    print("Finding optimal number of clusters...")
    K, inertias = clustering_model.get_optimal_clusters(umap_embeddings)
    visualizer.plot_elbow_curve(K, inertias)

    # Step 5: Apply clustering
    print("Applying K-means clustering...")
    clusters = clustering_model.apply_kmeans(umap_embeddings)
    visualizer.plot_kmeans_clusters(umap_embeddings, clusters, NUM_CLUSTERS)

    # Step 6: Apply DBSCAN
    print("Applying DBSCAN clustering...")
    dbscan_clusters = clustering_model.apply_dbscan(umap_embeddings)
    visualizer.plot_dbscan_clusters(umap_embeddings, dbscan_clusters)

    # Step 7: Extract themes
    print("Extracting themes...")
    theme_results = theme_extractor.extract_themes(sentences)
    print("\nTop words per theme:")
    print(theme_results['top_words'])

    # Step 8: Evaluate results
    print("\nEvaluating results...")
    visualizer.plot_confusion_matrix('designed_architecture')
    visualizer.plot_confusion_matrix('tfidf')
    visualizer.plot_metrics_comparison()

if __name__ == "__main__":
    main() 