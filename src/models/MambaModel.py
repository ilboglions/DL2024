import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model


class MambaModel(torch.nn.Module):
    def __init__(self, answer_tokens, model_size = '130m', cache_dir='models', LoRAConfig = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', cache_dir = cache_dir)
        self.tokenizer.pad_token = "<|padding|>"
        self.model = AutoModelForCausalLM.from_pretrained(f'state-spaces/mamba-{model_size}-hf', cache_dir=cache_dir)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenIndexes = [self.tokenizer.encode(x)[-1] for x in answer_tokens]
        print(self.tokenIndexes)

        if LoRAConfig is not None:
            self.model = get_peft_model(self.model, LoRAConfig)
    
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids,attention_mask)
        output.logits = output.logits[:,-1,:]
        if labels is not None:
            output.loss = self.criterion(output.logits, labels)
        output.logits = output.logits[:, self.tokenIndexes]
        return output

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_mapping(self):
        return self.tokenIndexes
    