import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput
from peft import get_peft_model



class OPTModel(torch.nn.Module):
    def __init__(self, answer_tokens, model_size = '125m', cache_dir='models', LoRAConfig = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(f'facebook/opt-{model_size}', cache_dir = cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(f'facebook/opt-{model_size}', cache_dir=cache_dir)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenIndexes = [self.tokenizer.encode(x)[-1] for x in answer_tokens]

        if LoRAConfig is not None:
            self.model = get_peft_model(self.model, LoRAConfig)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids,attention_mask)

        output = CausalLMOutput(
            loss=output.loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
        output.logits = output.logits[:,-1,:]
        if labels is not None:
            output.loss = self.criterion(output.logits, labels)
        output.logits = output.logits[:, self.tokenIndexes]  
        return output

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_mapping(self):
        return self.tokenIndexes
    
