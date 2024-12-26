import argparse
from utils import *
import wandb
from datetime import datetime

from contamination.time_travel import PROMPTS, eval_tt

def main(config):
    timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb.init(
    #     project='DeepLearning',  
    #     name=f"{config['model']['model_name']}_{config['model']['model_size']}_contam_{timestamp}"
    # )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['repo'],
        cache_dir=config['model']['cache_dir'],
        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['repo'],
        cache_dir=config['model']['cache_dir'],
        trust_remote_code=True
    ).to(device)

    general_datasets = {
        dataset['dataset_name'] : get_dataset(
            name=dataset['dataset_name'],
            tokenizer=tokenizer,
            answer_tokens=None,
            prompt=PROMPTS['general'],
            split=dataset['dataset_split'],
            cache_dir=config['dataset']['cache_dir']
        ) for dataset in config['dataset']['test'][:1]}
    guided_datasets = {
        dataset['dataset_name'] : get_dataset(
            name=dataset['dataset_name'],
            tokenizer=tokenizer,
            answer_tokens=None,
            prompt=PROMPTS['guided'],
            split=dataset['dataset_split'],
            cache_dir=config['dataset']['cache_dir'],
        ) for dataset in config['dataset']['test'][:1]}

    eval_tt(model, tokenizer, general_datasets, guided_datasets, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)