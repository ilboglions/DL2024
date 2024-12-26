import argparse
from utils import *
import wandb
from datetime import datetime

from contamination.tt import evaluate_tt


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

    if config['checker'] == 'time_travel':
        evaluate_tt(model, tokenizer, config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)