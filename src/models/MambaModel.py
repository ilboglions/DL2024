import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



class MambaModel(torch.nn.Module):
    def __init__(self, model_size = '130m', cache_dir='models'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', cache_dir = cache_dir)
        self.model = AutoModel.from_pretrained(f'state-spaces/mamba-{model_size}-hf', cache_dir=cache_dir)
        self.score = nn.Linear(self.model.config.hidden_size, 2)
    
    def forward(self, x):
        x = self.model(x)  
        x = self.score(x)
        return x

    def get_tokenizer(self):
        return self.tokenizer
    