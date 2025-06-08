import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
COMPLAINTS_FILE = os.path.join(DATA_DIR, 'complaints_2019_BofA.csv')
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'sentence_embeddings.csv')
UMAP_CLUSTERS_FILE = os.path.join(DATA_DIR, 'umap_embeddings_with_clusters_and_docs_k2.csv')

# Model parameters
MODEL_NAME = 'xlnet-large-cased'
MAX_LENGTH = 512
NUM_CLUSTERS = 2
NUM_THEMES = 2

# UMAP parameters
UMAP_PARAMS = {
    'n_neighbors': 10,
    'n_components': 2,
    'metric': 'euclidean'
}

# Clustering parameters
KMEANS_PARAMS = {
    'n_clusters': NUM_CLUSTERS,
    'random_state': 42
}

# TF-IDF parameters
TFIDF_PARAMS = {
    'max_df': 0.30,
    'token_pattern': r'\b[a-zA-Z]+\b'
}

# Stop words
CUSTOM_STOP_WORDS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'bank', 'america', 'boa', 'bofa', '///////', 'XXXX',
    'Me', 'My', 'xx', 'xxx', 'xxxx', 'xxxxx', 'cfpb',
    'bkofamerica', 'cfcb', ' xxx', ' xxxxx', 'cfpd', 'ba',
    'deposited', 'checks'
]

# Evaluation metrics
EVALUATION_METRICS = {
    'designed_architecture': {
        'true_negative': 17,
        'false_positive': 11,
        'false_negative': 8,
        'true_positive': 14
    },
    'tfidf': {
        'true_negative': 9,
        'false_positive': 16,
        'false_negative': 15,
        'true_positive': 10
    }
} 