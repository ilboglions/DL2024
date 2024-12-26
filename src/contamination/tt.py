import argparse

import wandb
import pandas as pd
from tqdm import tqdm

from utils import get_dataset
from contamination.time_travel.evaluation_phase import Alg1EvalPhase
from contamination.time_travel.metric_helper import Rouge, Bleurt

PROMPTS = {
    "general": """INSTRUCTION:
Finish SENTENCE 2 based on SENTENCE 1, such that the following LABEL shows the logical relationship between SENTENCE 1 and SENTENCE 2.

SENTENCE 1:
{premise}

LABEL: {label}

SENTENCE 2:
""",
    "guided": """INSTRUCTION:
You are provided with SENTENCE 1 from the {split} split of the {dataset} dataset.
Finish SENTENCE 2 as appeared in the dataset.
SENTENCE 2 MUST EXACTLY match the instance in the dataset.

SENTENCE 1:
{premise}

LABEL: {label}

SENTENCE 2:
""",
}


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


def generate(model, tokenizer, dataset, device):
    completions = []

    for sample in tqdm(dataset):
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=32,
            eos_token_id=tokenizer.eos_token_id)

        # Only decode the new tokens.
        new_tokens = outputs[:, input_ids.shape[-1]:]
        text_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        completions.extend(text_outputs)

    return completions


def evaluate_tt(model, tokenizer, config, device):
    general_datasets = fetch_datasets(tokenizer, PROMPTS['general'], config)
    guided_datasets = fetch_datasets(tokenizer, PROMPTS['guided'], config)

    # For each dataset, we generate general/guided completions and then eval Bleurt/Rouge
    for dataset_name in general_datasets.keys():
        print(f"Preprocessing dataset {dataset_name} ...")
        ds = general_datasets[dataset_name]
        completions = ds.dataset[ds.column_names[1]][:10] # get the hypothesis

        general_datasets[dataset_name].preprocess()
        guided_datasets[dataset_name].preprocess()

        print(f"Generating general completions for {dataset_name} ...")
        general_completions = generate(model, tokenizer, general_datasets[dataset_name], device)
    
        print(f"Generating guided completions for {dataset_name} ...")
        guided_completions = generate(model, tokenizer, guided_datasets[dataset_name], device)

        # Setup args/df expected by time_travel code
        # DF has 3 columns: completion, generated_general_completion, generated_guided_completion
        args = argparse.Namespace(
            experiment=f'results/time_travel/{dataset_name}',
            filepath=f'results/time_travel/{dataset_name}/df.csv',
            task='nli', 
            text_column=('', 'completion') # nli just uses the second element.
        )
        df = pd.DataFrame({
            'completion': completions,
            'generated_general_completion': general_completions,
            'generated_guided_completion': guided_completions
        })

        print(f"Evaluating Rouge for {dataset_name} ...")
        df = Alg1EvalPhase(
            df=df,
            args=args,
            scoring_tool=Rouge("rougeL"),
            save_intermediate_results=True,
        ).evaluate()

        print(f"Evaluating Bleurt for {dataset_name} ...")
        df = Alg1EvalPhase(
            df=df,
            args=args,
            scoring_tool=Bleurt(),
            save_intermediate_results=True,
        ).evaluate()
