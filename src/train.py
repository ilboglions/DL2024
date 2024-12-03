import argparse
from utils import *
from models.OPTModel import OPTModel
from models.MambaModel import MambaModel
from data.NLIDataset import *
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    args = parser.parse_args()
    config = load_config(args.config)

    print(config)
    

if __name__ == "__main__":
    main()