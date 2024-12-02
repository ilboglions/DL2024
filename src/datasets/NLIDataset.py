from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np

class NLIDataset(Dataset):

    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.preprocess()

    def tokenize_function(self, samples):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def change_labels(self, samples):
        raise NotImplementedError("Subclasses should implement this method.")

    def preprocess(self):
        self.dataset = self.dataset.map(self.tokenize_function, batched=True) 
        self.dataset = self.dataset.map(self.change_labels, batched=True)
        self.dataset = self.dataset.rename_column("label", "labels") 
        #self.dataset = self.dataset.select_columns(["input_ids", "attention_mask", "labels"])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        tokenized_data = self.dataset[idx]
        input_ids = tokenized_data['input_ids']
        attention_mask = tokenized_data['attention_mask']
        labels = tokenized_data['labels']
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
class SNLIDataset(NLIDataset):
    def __init__(self, tokenizer, dataset_name="stanfordnlp/snli", split='train'):
        dataset = load_dataset(dataset_name, cache_dir="datasets", split=split)
        super().__init__(tokenizer, dataset)

    def preprocess(self):
        print(f"Processing SNLI-style Hugging Face dataset")
        self.dataset = self.dataset.filter(lambda example: example["label"] != -1 and example["label"] != 1)
        super().preprocess()
        
    def change_labels(self,batch):
        labels = np.array(batch['label'])  
        labels[labels == 2] = 1  
        batch['label'] = labels.tolist() 
        return batch
     
    def tokenize_function(self,batch):
        return self.tokenizer(
        batch["premise"], batch["hypothesis"], truncation=True, padding="max_length", max_length=128
        )
    
class SciTailDataset(NLIDataset):
    def __init__(self, tokenizer, dataset_name="allenai/scitail", format= "dgem_format", split='train'):
        dataset = load_dataset(dataset_name, format,cache_dir="datasets", split=split)
        super().__init__(tokenizer, dataset)

    def preprocess(self):
        print(f"Processing SciTail-style Hugging Face dataset")
        self.dataset = self.dataset.filter(lambda example: example["label"] != 'neutral')
        super().preprocess()
        
    def change_labels(self,batch):
        values = np.zeros(len(batch['label']), dtype=int)
        values[batch['label'] == 'entails'] = 0 
        values[batch['label'] == 'contradiction'] = 1
        batch['label'] = values
        return batch

    def tokenize_function(self,batch):
        return self.tokenizer(
        batch["premise"], batch["hypothesis"], truncation=True, padding="max_length", max_length=128
        )

class GLUEDataset(NLIDataset):
    def __init__(self, tokenizer, dataset_name="nyu-mll/glue", subset= "wnli", split='train'):
        dataset = load_dataset(dataset_name, subset ,cache_dir="datasets", split=split)
        self.subset = subset
        super().__init__(tokenizer, dataset)

    def preprocess(self):
        print(f"Processing GLUE-style Hugging Face dataset")
        super().preprocess()

    def change_labels(self,batch):
        if self.subset == 'wnli':
            values = np.zeros(len(batch['label']), dtype=int)
            values[np.where(np.array(batch['label']) == 0)[0]] = 1 
            values[np.where(np.array(batch['label']) == 1)[0]] = 0
            batch['label'] = values
            print(batch['label'])
        return batch  

    def tokenize_function(self,batch):
        return self.tokenizer(
        batch["sentence1"], batch["sentence2"], truncation=True, padding="max_length", max_length=128
        )

class PAWSDataset(NLIDataset):
    def __init__(self, tokenizer, dataset_name="google-research-datasets/paws", subset= "labeled_final", split='train'):
        dataset = load_dataset(dataset_name, subset ,cache_dir="datasets", split=split)
        super().__init__(tokenizer, dataset)

    def preprocess(self):
        print(f"Processing PAWS-style Hugging Face dataset")
        super().preprocess()

    def change_labels(self,batch):
        values = np.zeros(len(batch['label']), dtype=int)
        values[np.where(np.array(batch['label']) == 0)[0]] = 1 
        values[np.where(np.array(batch['label']) == 1)[0]] = 0
        batch['label'] = values
        print(batch['label'])
        return batch
        
    def tokenize_function(self,batch):
        return self.tokenizer(
        batch["sentence1"], batch["sentence2"], truncation=True, padding="max_length", max_length=128
        )

class HANSDataset(NLIDataset):
    def __init__(self, tokenizer, dataset_name="jhu-cogsci/hans", split='train'):
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        super().__init__(tokenizer, dataset)

    def preprocess(self):
        print(f"Processing Hans-style Hugging Face dataset")
        super().preprocess()
        
    def change_labels(self,batch):
        return batch
     
    def tokenize_function(self,batch):
        return self.tokenizer(
        batch["premise"], batch["hypothesis"], truncation=True, padding="max_length", max_length=128
        )

class ANLIDataset(NLIDataset):
    def __init__(self, tokenizer, dataset_name="facebook/anli", split='train_r3'):
        dataset = load_dataset(dataset_name, split=split)
        super().__init__(tokenizer, dataset)

    def preprocess(self):
        print(f"Processing anli-style Hugging Face dataset")
        self.dataset = self.dataset.filter(lambda example: example["label"] != 1)
        super().preprocess()
        
    def change_labels(self,batch):
        labels = np.array(batch['label'])  
        labels[labels == 2] = 1  
        batch['label'] = labels.tolist() 
        return batch
     
    def tokenize_function(self,batch):
        return self.tokenizer(
        batch["premise"], batch["hypothesis"], truncation=True, padding="max_length", max_length=128
        )



    
