import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import get_peft_model
from transformers.modeling_outputs import CausalLMOutput


class MambaModel(torch.nn.Module):
    def __init__(self, answer_tokens, model_size = '130m', cache_dir='models', LoRAConfig = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(f'AntonV/mamba2-{model_size}-hf', cache_dir = cache_dir)
        config = AutoConfig.from_pretrained(f'AntonV/mamba2-{model_size}-hf', cache_dir=cache_dir)
        config.num_groups = torch.cuda.device_count()
        self.model = AutoModelForCausalLM.from_pretrained(f'AntonV/mamba2-{model_size}-hf', cache_dir=cache_dir, config=config)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenIndexes = [self.tokenizer.encode(x)[-1] for x in answer_tokens]

        if LoRAConfig is not None:
            self.model = get_peft_model(self.model, LoRAConfig)
    
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids,attention_mask=attention_mask)
        logits = output.logits[:,-1,:]
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        logits = logits[:, self.tokenIndexes]
        return CausalLMOutput(loss=loss, logits=logits)

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_mapping(self):
        return self.tokenIndexes
    