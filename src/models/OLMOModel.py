import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput
from peft import get_peft_model



class OLMOModel(torch.nn.Module):
    def __init__(self, answer_tokens, model_size = '1B', cache_dir='models', LoRAConfig = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(f'allenai/OLMo-{model_size}-0724-hf', cache_dir = cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(f'allenai/OLMo-{model_size}-0724-hf', cache_dir=cache_dir)
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
    
