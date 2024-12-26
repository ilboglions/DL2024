from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset
import numpy as np
import torch


class NLIDataset(Dataset):

    def __init__(self, tokenizer, padding, answer_tokens, prompt, dataset, preprocess=True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.dataset = dataset
        self.answer_tokens = answer_tokens
        self.prompt = prompt

        if preprocess:
            self.preprocess()

    def tokenize_function(self, batch):
        premise, hypothesis, label = self.column_names
        combined = self.get_combined_string(batch[premise], batch[hypothesis], batch[label])
        return self.tokenizer(
            combined, truncation=True, padding=self.padding, max_length=128)   
    
    def get_combined_string(self, premise, hypothesis, labels):
        premise = np.array(premise)
        hypothesis = np.array(hypothesis)
        label = np.array(labels).astype(str)
        placeholders = {
            "premise": premise, 
            "hypothesis": hypothesis, 
            "label": label,
            "dataset": self.dataset.info.dataset_name,
            "split": str(self.dataset.split),
        }
    
        combined = np.full(premise.shape, self.prompt)
        for placeholder, values in placeholders.items():
            combined = np.char.replace(combined, f"{{{placeholder}}}", values)
        return combined.tolist()

    def change_labels(self, samples):
        raise NotImplementedError('Subclasses should implement this method.')
    
    def preprocess(self):
        self.dataset = self.dataset.map(self.tokenize_function, batched=True) 
        if self.answer_tokens is not None:
            self.dataset = self.dataset.map(self.change_labels, batched=True)
        self.dataset = self.dataset.rename_column('label', 'labels') 
        self.dataset = self.dataset.select_columns(['input_ids', 'attention_mask', 'labels'])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        tokenized_data = self.dataset[idx]
        input_ids = tokenized_data['input_ids']
        attention_mask = tokenized_data['attention_mask']

        if self.answer_tokens is not None:
            labels = tokenized_data['labels']
            return {"input_ids" : torch.tensor(input_ids), 
                    "attention_mask" : torch.tensor(attention_mask), 
                    "labels" : torch.tensor(labels)}
        else:
            return {"input_ids" : torch.tensor(input_ids), 
                    "attention_mask" : torch.tensor(attention_mask)}


class SNLIDataset(NLIDataset):
    dataset_name = 'stanfordnlp/snli'
    column_names = ('premise', 'hypothesis', 'label')

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, cache_dir=cache_dir, split=split)
        super().__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)

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
    

class SciTailDataset(NLIDataset):
    dataset_name = 'allenai/scitail'
    fmt = 'dgem_format'
    column_names = ('premise', 'hypothesis', 'label')

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, self.fmt, cache_dir=cache_dir, split=split)
        super().__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)

    def preprocess(self):
        print(f'Processing SciTail-style Hugging Face dataset')
        super().preprocess()
        
    def change_labels(self,batch):
        values = np.zeros(len(batch['label']), dtype=int)
        values[np.where(np.array(batch['label']) == 'entails')] = self.tokenizer.encode(self.answer_tokens[0])[-1]
        values[np.where(np.array(batch['label']) == 'neutral')] = self.tokenizer.encode(self.answer_tokens[1])[-1]  
        batch['label'] = values
        return batch


class GLUEDataset(NLIDataset):
    dataset_name = 'nyu-mll/glue'
    subset = 'wnli'
    column_names = ('sentence1', 'sentence2', 'label')

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, self.subset, cache_dir=cache_dir, split=split)
        super().__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)

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


class RTEDataset(GLUEDataset):
    dataset_name = 'nyu-mll/glue'
    subset = 'rte'

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, self.subset, cache_dir=cache_dir, split=split)
        super(GLUEDataset, self).__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)

class WNLIDataset(GLUEDataset):
    dataset_name = 'nyu-mll/glue'
    subset = 'wnli'

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, self.subset, cache_dir=cache_dir, split=split)
        super(GLUEDataset, self).__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)
    
    def change_labels(self, batch):
        values = np.zeros(len(batch['label']), dtype=int)
        values[np.where(np.array(batch['label']) == 0)[0]] = self.tokenizer.encode(self.answer_tokens[1])[-1] 
        values[np.where(np.array(batch['label']) == 1)[0]] = self.tokenizer.encode(self.answer_tokens[0])[-1] 
        batch['label'] = values
        return batch

class MNLIDataset(SNLIDataset):
    dataset_name='nyu-mll/glue'
    subset = 'mnli'

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, self.subset, cache_dir=cache_dir, split=split)
        super(SNLIDataset, self).__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)

class PAWSDataset(NLIDataset):
    dataset_name = 'google-research-datasets/paws'
    subset = 'labeled_final'
    column_names = ('sentence1', 'sentence2', 'label')

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, self.subset, cache_dir=cache_dir, split=split)
        super().__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)

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


class HANSDataset(NLIDataset):
    dataset_name = 'jhu-cogsci/hans'
    column_names = ('premise', 'hypothesis', 'label')

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, split=split, trust_remote_code=True, cache_dir=cache_dir)
        super().__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)

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


class ANLIDataset(NLIDataset):
    dataset_name='facebook/anli'
    column_names = ('premise', 'hypothesis', 'label')

    def __init__(self, tokenizer, padding, answer_tokens, prompt, split='train_r3', cache_dir='datasets', preprocess=True):
        dataset = load_dataset(self.dataset_name, split=split, cache_dir=cache_dir)
        super().__init__(tokenizer, padding, answer_tokens, prompt, dataset, preprocess=preprocess)

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
