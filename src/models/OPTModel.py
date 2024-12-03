import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer



class OPTModel(torch.nn.Module):
    def __init__(self, model_size = '125m', cache_dir='models'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(f'facebook/opt-{model_size}', cache_dir = cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(f'facebook/opt-{model_size}', cache_dir=cache_dir)
        self.model = model.model
        self.score = model.score 
    
    def forward(self, x):
        x = self.model(x)
        x = self.score(x)     
        return x

    def get_tokenizer(self):
        return self.tokenizer
    