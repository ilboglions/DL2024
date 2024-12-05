import argparse
from utils import *
from peft import LoraConfig
from transformers import Trainer, TrainingArguments
import wandb

def main(config):

    LoRAConfig = None
    if config['model']['LoRA']:
        LoRAConfig = LoraConfig(**config['model']['LoRA'])

    wandb.init(
        project='DeepLearning',  
        name=f"{config['model']['model_name']}_{config['model']['model_size']}_{config['dataset']['train']['dataset_name']}_{config['model']['training']['seed']}"
    )

    model = get_model(config['model']['model_name'], config['model']['answer_tokens'], LoRAConfig=LoRAConfig)
    train_dataset = get_dataset(config['dataset']['train']['dataset_name'], model.get_tokenizer(), config['model']['answer_tokens'], split=config['dataset']['train']['dataset_split'])
    eval_datasets = {dataset['dataset_name'] : get_dataset(dataset['dataset_name'], model.get_tokenizer(), config['model']['answer_tokens'], split=dataset['dataset_split']) for dataset in config['dataset']['test']}

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
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)