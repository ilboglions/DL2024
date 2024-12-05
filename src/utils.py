import tomli
from data.NLIDataset import *
from models.MambaModel import *
from models.OPTModel import *
from sklearn.metrics import accuracy_score, f1_score
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

def get_dataset(name, *args, **kwargs):
    
    return DATASET_DICT[name](*args, **kwargs)
    
def get_model(model_name, *args, **kwargs):

    return MODEL_DICT[model_name](*args, **kwargs)

def eval_metrics(eval_predictions, mapping):
    pred = eval_predictions.predictions
    labels = eval_predictions.label_ids
    bin_labels = np.zeros(labels.shape)
    bin_labels[labels == mapping[1]] = 1
    pred_labels = np.argmax(pred, axis=1)
    return {"accuracy" : accuracy_score(bin_labels, pred_labels),
            "f1" : f1_score(bin_labels, pred_labels)}

