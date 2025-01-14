import argparse
from utils import *
from peft import LoraConfig
from transformers import Trainer, TrainingArguments
import wandb
import deepspeed
from datetime import datetime

def main(config, local_rank):

    LoRAConfig = None
    if config['model']['LoRA']:
        LoRAConfig = LoraConfig(**config['model']['LoRA'])

    timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

    if local_rank <= 0 and config['model']['training']['report_to'] == 'wandb':
        wandb.init(
            project='DeepLearning',  
            name=f"{config['model']['model_name']}_{config['model']['model_size']}_{config['dataset']['train']['dataset_name']}_{config['model']['training']['seed']}_{timestamp}"
        )

    model = get_model(config['model']['model_name'], config['model']['answer_tokens'], model_size=config['model']['model_size'],cache_dir=config['model']['cache_dir'] ,LoRAConfig=LoRAConfig)
    train_dataset = get_dataset(
        name=config['dataset']['train']['dataset_name'],
        tokenizer=model.get_tokenizer(),
        padding='max_length',
        answer_tokens=config['model']['answer_tokens'],
        prompt=config['model']['prompt'],
        split=config['dataset']['train']['dataset_split'])
    eval_datasets = {
        dataset['dataset_name'] : get_dataset(
            name=dataset['dataset_name'],
            tokenizer=model.get_tokenizer(),
            padding='max_length',
            answer_tokens=config['model']['answer_tokens'],
            prompt=config['model']['prompt'],
            split=dataset['dataset_split']
        ) for dataset in config['dataset']['test']}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**config['model']['training']),
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_metrics=lambda eval_predictions: eval_metrics(eval_predictions, model.get_mapping())
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config, args.local_rank)