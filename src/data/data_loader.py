import pandas as pd
import torch
from transformers import XLNetTokenizer
from src.config.config import MODEL_NAME, MAX_LENGTH

class DataLoader:
    def __init__(self, device):
        self.device = device
        self.tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)

    def load_complaints(self, file_path):
        """Load complaints from CSV file."""
        df = pd.read_csv(file_path)
        return df['Consumer complaint narrative'].tolist()

    def get_embeddings(self, sentences, model):
        """Generate embeddings for a list of sentences using XLNet."""
        embeddings = []
        for sentence in sentences:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            mean_embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()
            embeddings.append(mean_embedding)
        
        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """Save embeddings to numpy and CSV files."""
        import numpy as np
        embeddings_array = np.array(embeddings)
        np.save(output_path.replace('.csv', '.npy'), embeddings_array)
        
        embeddings_df = pd.DataFrame(embeddings_array)
        embeddings_df.to_csv(output_path, index=False)

    def load_embeddings(self, file_path):
        """Load embeddings from CSV file."""
        return pd.read_csv(file_path).values

    def prepare_cluster_data(self, complaints_df, clusters_df):
        """Prepare data for clustering analysis."""
        complaints_df['Document_ID'] = complaints_df.index + 1
        merged_df = pd.merge(complaints_df, clusters_df, on='Document_ID', how='left')
        merged_df.columns.values[0] = 'ComplaintTxt'
        return merged_df

    def concatenate_by_cluster(self, df):
        """Concatenate complaints by cluster."""
        df['Text'] = df['Text'].astype('string')
        return df.groupby('Cluster')['Text'].agg('///////'.join).reset_index() 