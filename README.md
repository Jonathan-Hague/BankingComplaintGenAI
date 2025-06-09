# Customer Complaint Analysis and Theme Extraction

This project implements a deep learning-based approach for analyzing customer complaints and extracting themes using XLNet embeddings, UMAP dimensionality reduction, and various clustering techniques. The system processes customer complaints, generates embeddings, performs clustering analysis, and extracts meaningful themes from the data.

## Project Overview

The project uses a combination of advanced NLP and machine learning techniques:
- XLNet for generating contextual embeddings
- UMAP for dimensionality reduction
- Multiple clustering algorithms (K-means, DBSCAN, HDBSCAN)
- TF-IDF and NMF for theme extraction
- Comprehensive visualization tools

## Project Structure

```
.
├── src/
│   ├── config/
│   │   └── config.py           # Configuration settings and parameters
│   ├── data/
│   │   └── data_loader.py      # Data loading and preprocessing utilities
│   ├── models/
│   │   ├── embedding_model.py  # XLNet embedding model implementation
│   │   ├── clustering.py       # Clustering algorithms implementation
│   │   └── theme_extractor.py  # Theme extraction using TF-IDF and NMF
│   ├── visualization/
│   │   └── visualizer.py       # Visualization utilities and plotting functions
│   └── main.py                 # Main execution script
├── data/                       # Data directory for input/output files
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Component Details

### 1. Configuration (`src/config/config.py`)
- Contains all configurable parameters
- File paths and directories
- Model parameters
- Clustering settings
- Stop words and custom configurations
- Evaluation metrics

### 2. Data Loading (`src/data/data_loader.py`)
- Handles data loading from CSV files
- Preprocesses text data
- Generates embeddings using XLNet
- Saves and loads embeddings
- Prepares data for clustering

### 3. Models
#### Embedding Model (`src/models/embedding_model.py`)
- Implements XLNet model for text embeddings
- Handles GPU/CPU device management
- Provides embedding generation functionality

#### Clustering (`src/models/clustering.py`)
- Implements multiple clustering algorithms:
  - K-means clustering
  - DBSCAN clustering
  - HDBSCAN clustering
- Includes UMAP dimensionality reduction
- Provides optimal cluster number detection

#### Theme Extractor (`src/models/theme_extractor.py`)
- Implements TF-IDF vectorization
- Performs Non-negative Matrix Factorization (NMF)
- Extracts themes from documents
- Groups documents by themes

### 4. Visualization (`src/visualization/visualizer.py`)
- Generates various visualizations:
  - t-SNE/UMAP plots
  - Clustering results
  - Elbow curves
  - Confusion matrices
  - Performance metrics comparison

### 5. Main Script (`src/main.py`)
- Orchestrates the entire pipeline
- Handles the execution flow
- Manages model initialization
- Coordinates data processing steps

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the data:
   - Place the complaint data file (`complaints_2019_BofA.csv`) in the `data` directory
   - Ensure the CSV file has a 'Consumer complaint narrative' column

2. Run the analysis:
```bash
python src/main.py
```

The script will execute the following steps:
1. Load and preprocess the complaints
2. Generate embeddings using XLNet
3. Apply UMAP dimensionality reduction
4. Perform clustering analysis
5. Extract themes
6. Generate visualizations
7. Evaluate results

## Output Files

The script generates several output files in the `data` directory:
- `sentence_embeddings.npy`: Raw embeddings
- `sentence_embeddings.csv`: Embeddings in CSV format
- `umap_embeddings_with_clusters_and_docs_k2.csv`: Clustering results
- Various visualization plots

## Configuration

The following prameters in `src/config/config.py` can be modified:

1. Model Parameters:
   - `MODEL_NAME`: XLNet model variant
   - `MAX_LENGTH`: Maximum sequence length
   - `NUM_CLUSTERS`: Number of clusters for K-means

2. Clustering Parameters:
   - UMAP parameters (n_neighbors, n_components)
   - K-means parameters
   - DBSCAN parameters

3. Data Processing:
   - Stop words
   - Custom filters
   - File paths

## Dependencies

- Python 3.7+
- PyTorch
- Transformers
- scikit-learn
- UMAP-learn
- HDBSCAN
- pandas
- numpy
- matplotlib
- seaborn
- nltk

## Performance Metrics

The project evaluates two approaches:
1. XLNet-based Designed Architecture
   - Accuracy: 0.62
   - Precision: 0.56
   - Recall: 0.64

2. Baseline TF-IDF
   - Accuracy: 0.38
   - Precision: 0.38
   - Recall: 0.40

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact

jhague@stanford.edu
