import tomli
from data.NLIDataset import *
from models.MambaModel import *
from models.OPTModel import *

DATASET_DICT = {
    'snli' : SNLIDataset,
    'mnli' : MNLIDataset,
    'scitail': SciTailDataset,
    'wnli': WNLIDataset,
    'rte': RTEDataset,
    'paws': PAWSDataset,
    'hans': HANSDataset,
    'anli': ANLIDataset
}

MODEL_DICT = {
    'opt' : OPTModel,
    'mamba': MambaModel
}


def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)

def get_dataset(name, tokenizer, split, cache_dir = 'datasets'):
    
    return DATASET_DICT[name](tokenizer=tokenizer, split=split, cache_dir=cache_dir)
    
def get_model(model_name, model_size, cache_dir='models'):

    return MODEL_DICT[model_name](model_size=model_size, cache_dir = cache_dir)