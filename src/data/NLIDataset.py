from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset
import numpy as np
import torch

class NLIDataset(Dataset):

    def __init__(self, tokenizer, answer_tokens, prompt, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        print(self.dataset.info)
        self.answer_tokens = answer_tokens
        self.prompt = prompt
        self.preprocess()

    def tokenize_function(self, samples):
        raise NotImplementedError('Subclasses should implement this method.')
    
    def get_combined_string(self, premise, hypothesis, labels):
        premise = np.array(premise)
        hypothesis = np.array(hypothesis)
        label = np.array(labels).astype(str)
        placeholders = {
            "premise": premise, 
            "hypothesis": hypothesis, 
            "label": label,
            "dataset": self.dataset.info.dataset_name,
            "split": self.dataset.split,
        }
    
        combined = np.full(premise.shape, self.prompt)
        for placeholder, values in placeholders.items():
            combined = np.char.replace(combined, f"{{{placeholder}}}", values)
        return combined.tolist()

    def change_labels(self, samples):
        raise NotImplementedError('Subclasses should implement this method.')
    
    def preprocess(self):
        self.dataset = self.dataset.map(self.tokenize_function, batched=True) 
        self.dataset = self.dataset.map(self.change_labels, batched=True)
        self.dataset = self.dataset.rename_column('label', 'labels') 
        self.dataset = self.dataset.select_columns(['input_ids', 'attention_mask', 'labels'])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        tokenized_data = self.dataset[idx]
        input_ids = tokenized_data['input_ids']
        attention_mask = tokenized_data['attention_mask']
        labels = tokenized_data['labels']
        return {"input_ids" : torch.tensor(input_ids), 
                "attention_mask" : torch.tensor(attention_mask), 
                "labels" : torch.tensor(labels)}
    
class SNLIDataset(NLIDataset):

    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='stanfordnlp/snli', split='train', cache_dir='datasets'):
        dataset = load_dataset(dataset_name, cache_dir=cache_dir, split=split)
        super().__init__(tokenizer, answer_tokens, prompt, dataset)

    def preprocess(self):
        print(f'Processing SNLI-style Hugging Face dataset')
        self.dataset = self.dataset.filter(lambda example: example['label'] != -1 and example['label'] != 1)
        new_features = self.dataset.features.copy()
        new_features["label"] = ClassLabel(num_classes=self.tokenizer.vocab_size)
        self.dataset = self.dataset.cast(new_features)
        super().preprocess()
        
    def change_labels(self,batch):
        labels = np.array(batch['label'])  
        labels[labels == 2] = self.tokenizer.encode(self.answer_tokens[1])[-1]
        labels[labels == 0] = self.tokenizer.encode(self.answer_tokens[0])[-1]
        batch['label'] = labels.tolist() 
        return batch
     
    def tokenize_function(self,batch):
        combined = self.get_combined_string(batch['premise'], batch['hypothesis'], batch['label'])
        return self.tokenizer(
        combined, truncation=True, padding='max_length', max_length=128
        )        

class SciTailDataset(NLIDataset):

    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='allenai/scitail', format= 'dgem_format', split='train', cache_dir = 'datasets'):
        dataset = load_dataset(dataset_name, format,cache_dir=cache_dir, split=split)
        super().__init__(tokenizer, answer_tokens, prompt, dataset)

    def preprocess(self):
        print(f'Processing SciTail-style Hugging Face dataset')
        super().preprocess()
        
    def change_labels(self,batch):
        values = np.zeros(len(batch['label']), dtype=int)
        values[np.where(np.array(batch['label']) == 'entails')] = self.tokenizer.encode(self.answer_tokens[0])[-1]
        values[np.where(np.array(batch['label']) == 'neutral')] = self.tokenizer.encode(self.answer_tokens[1])[-1]  
        batch['label'] = values
        return batch

    def tokenize_function(self,batch):
        combined = self.get_combined_string(batch['premise'], batch['hypothesis'], batch['label'])
        return self.tokenizer(
        combined, truncation=True, padding='max_length', max_length=128
        )  

class GLUEDataset(NLIDataset):

    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='nyu-mll/glue', subset= 'wnli', split='train', cache_dir='datasets'):
        dataset = load_dataset(dataset_name, subset ,cache_dir=cache_dir, split=split)
        super().__init__(tokenizer, answer_tokens, prompt, dataset)

    def preprocess(self):
        print(f'Processing GLUE-style Hugging Face dataset')
        new_features = self.dataset.features.copy()
        new_features["label"] = ClassLabel(num_classes=self.tokenizer.vocab_size)
        self.dataset = self.dataset.cast(new_features)
        super().preprocess()

    def change_labels(self,batch):
        labels = np.array(batch['label'])  
        labels[labels == 1] = self.tokenizer.encode(self.answer_tokens[1])[-1]  
        labels[labels == 0] = self.tokenizer.encode(self.answer_tokens[0])[-1]
        batch['label'] = labels.tolist() 
        return batch

    def tokenize_function(self,batch):
        combined = self.get_combined_string(batch['sentence1'], batch['sentence2'], batch['label'])
        return self.tokenizer(
        combined, truncation=True, padding='max_length', max_length=128
        )  
    
class RTEDataset(GLUEDataset):

    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='nyu-mll/glue', subset= 'rte', split='train', cache_dir='datasets'):
        dataset = load_dataset(dataset_name, subset ,cache_dir=cache_dir, split=split)
        super(GLUEDataset, self).__init__(tokenizer, answer_tokens, prompt, dataset)

class WNLIDataset(GLUEDataset):

    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='nyu-mll/glue', subset= 'wnli', split='train', cache_dir='datasets'):
        dataset = load_dataset(dataset_name, subset ,cache_dir=cache_dir, split=split)
        super(GLUEDataset, self).__init__(tokenizer, answer_tokens, prompt, dataset)
    
    def change_labels(self, batch):
        values = np.zeros(len(batch['label']), dtype=int)
        values[np.where(np.array(batch['label']) == 0)[0]] = self.tokenizer.encode(self.answer_tokens[1])[-1] 
        values[np.where(np.array(batch['label']) == 1)[0]] = self.tokenizer.encode(self.answer_tokens[0])[-1] 
        batch['label'] = values
        return batch

class MNLIDataset(SNLIDataset):
    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='nyu-mll/glue', subset= 'mnli',split='train', cache_dir='datasets'):
        dataset = load_dataset(dataset_name, subset,cache_dir=cache_dir, split=split)
        super(SNLIDataset, self).__init__(tokenizer, answer_tokens, prompt, dataset)

class PAWSDataset(NLIDataset):
    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='google-research-datasets/paws', subset= 'labeled_final', split='train', cache_dir='datasets'):
        dataset = load_dataset(dataset_name, subset ,cache_dir=cache_dir, split=split)
        super().__init__(tokenizer, answer_tokens, prompt, dataset)

    def preprocess(self):
        print(f'Processing PAWS-style Hugging Face dataset')
        new_features = self.dataset.features.copy()
        new_features["label"] = ClassLabel(num_classes=self.tokenizer.vocab_size)
        self.dataset = self.dataset.cast(new_features)
        super().preprocess()

    def change_labels(self,batch):
        values = np.zeros(len(batch['label']), dtype=int)
        values[np.where(np.array(batch['label']) == 0)[0]] = self.tokenizer.encode(self.answer_tokens[1])[-1] 
        values[np.where(np.array(batch['label']) == 1)[0]] = self.tokenizer.encode(self.answer_tokens[0])[-1] 
        batch['label'] = values
        return batch
        
    def tokenize_function(self,batch):
        combined = self.get_combined_string(batch['sentence1'], batch['sentence2'], batch['label'])
        return self.tokenizer(
        combined, truncation=True, padding='max_length', max_length=128
        )  

class HANSDataset(NLIDataset):
    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='jhu-cogsci/hans', split='train', cache_dir='datasets'):
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True, cache_dir=cache_dir)
        super().__init__(tokenizer, answer_tokens, prompt, dataset)

    def preprocess(self):
        print(f'Processing Hans-style Hugging Face dataset')
        new_features = self.dataset.features.copy()
        new_features["label"] = ClassLabel(num_classes=self.tokenizer.vocab_size)
        self.dataset = self.dataset.cast(new_features)
        super().preprocess()
        
    def change_labels(self,batch):
        labels = np.array(batch['label'])  
        labels[labels == 1] = self.tokenizer.encode(self.answer_tokens[1])[-1]  
        labels[labels == 0] = self.tokenizer.encode(self.answer_tokens[0])[-1]
        batch['label'] = labels.tolist() 
        return batch
     
    def tokenize_function(self,batch):
        combined = self.get_combined_string(batch['premise'], batch['hypothesis'], batch['label'])
        return self.tokenizer(
        combined, truncation=True, padding='max_length', max_length=128
        ) 

class ANLIDataset(NLIDataset):
    def __init__(self, tokenizer, answer_tokens, prompt, dataset_name='facebook/anli', split='train_r3', cache_dir='datasets'):
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        super().__init__(tokenizer, answer_tokens, prompt, dataset)

    def preprocess(self):
        print(f'Processing anli-style Hugging Face dataset')
        self.dataset = self.dataset.filter(lambda example: example['label'] != 1)
        new_features = self.dataset.features.copy()
        new_features["label"] = ClassLabel(num_classes=self.tokenizer.vocab_size)
        self.dataset = self.dataset.cast(new_features)
        super().preprocess()
        
    def change_labels(self,batch):
        labels = np.array(batch['label'])  
        labels[labels == 2] = self.tokenizer.encode(self.answer_tokens[1])[-1]
        labels[labels == 0] = self.tokenizer.encode(self.answer_tokens[0])[-1] 
        batch['label'] = labels.tolist() 
        return batch
     
    def tokenize_function(self,batch):
        combined = self.get_combined_string(batch['premise'], batch['hypothesis'], batch['label'])
        return self.tokenizer(
        combined, truncation=True, padding='max_length', max_length=128
        ) 
