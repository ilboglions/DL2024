import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from contamination.tt import evaluate_tt
from contamination.minkpp import evaluate_minkpp
from utils import load_config


def main(config):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

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
    elif config['checker'] == 'minkpp':
        evaluate_minkpp(model, tokenizer, config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)