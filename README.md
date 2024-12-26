# Deep Learning Project 2024 - ETH Zürich
### Matteo Boglioni, Francesco Rita, Andrea Sgobbi, Gabriel Tavernini
## Do Generalization Results Generalize?

## 1. Project Description
TODO

## 2. Repository Map
```
    .
    ├── README.md
    ├── requirements.txt
    ├── datasets
    ├── models
    ├── results
    └── src
        ├── __init__.py
        ├── datasets
        │   └── NLIDataset.py
        ├── models
        └── notebooks

```

## 3. Project Setup

Our code was developed with Python 3.11.10. Please make sure you have this specific version installed on your system.
Create a virtual/conda environment and install project's dependencies by running:
```bash
    pip install -r requirements.txt
```


### Contamination checks

```bash
git clone https://github.com/google-research/bleurt.git dependencies/bleurt_scorer
cd dependencies/bleurt_scorer
pip install .
```

```bash
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
```