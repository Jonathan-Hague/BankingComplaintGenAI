import torch
from transformers import XLNetModel
from src.config.config import MODEL_NAME

class EmbeddingModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XLNetModel.from_pretrained(MODEL_NAME).to(self.device)

    def get_device(self):
        return self.device

    def get_model(self):
        return self.model 