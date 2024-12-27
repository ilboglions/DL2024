import argparse

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils import get_dataset
from contamination.tt import PROMPTS


def fetch_datasets(tokenizer, prompt, config):
    return {
        dataset['dataset_name'] : get_dataset(
            name=dataset['dataset_name'],
            tokenizer=tokenizer,
            padding='do_not_pad',
            answer_tokens=None,
            prompt=prompt,
            split=dataset['dataset_split'],
            cache_dir=config['dataset']['cache_dir'],
            preprocess=False
        ) for dataset in config['dataset']['test']
    }


def compute_scores(model, tokenizer, dataset, device):
    scores = defaultdict(list)
    for sample in tqdm(dataset):
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        assert input_ids.shape[0] == 1
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        with torch.no_grad():
          outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        ll = -loss.item()
        
        # assuming the score is larger for training data
        # and smaller for non-training data
        # this is why sometimes there is a negative sign in front of the score
        
        # loss and zlib
        scores['loss'].append(ll)
        scores['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))

        # mink and mink++
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        ## mink
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            scores[f'mink_{ratio}'].append(np.mean(topk).item())

        ## mink++
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            scores[f'mink++_{ratio}'].append(np.mean(topk).item())
    return scores

def evaluate_minkpp(model, tokenizer, config, device):
    general_datasets = fetch_datasets(tokenizer, PROMPTS['general'], config)

    # For each dataset, we generate general/guided completions and then eval Bleurt/Rouge
    for dataset_name in general_datasets.keys():
        print(f"Preprocessing dataset {dataset_name} ...")
        general_datasets[dataset_name].preprocess()

        print(f"Compute scores for {dataset_name} ...")
        scores = compute_scores(model, tokenizer, general_datasets[dataset_name], device)

        print(f"Compute labels from scores {dataset_name} ...")
        df = pd.DataFrame()
        for key, values in scores.items():
            df[key] = values
            df[f"L_{key}"] = [1 if value >= 0.5 else 0 for value in values]

        summary = pd.DataFrame({
            'avg': df.mean(),
            'min': df.min(),
            'max': df.max()
        })
        print(summary)

        save_root = f"results/{dataset_name}"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        model_id = config['model']['repo'].split('/')[-1]
        df.to_csv(os.path.join(save_root, f"{model_id}.csv"), index=False)
        
