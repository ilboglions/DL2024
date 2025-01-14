# Do Generalization Results Generalize?
### Matteo Boglioni, Francesco Rita, Andrea Sgobbi, Gabriel Tavernini

## 1. Project Description
This project aims to investigate out-of-domain (OOD) generalization in the context of Natural Language Inference. We assess the performance of 3 LLM families: Meta's OPT, CMU's Mamba and AI2's OLMo with parameter scales ranging from 125M to 7B. Key contributions of this project consist in a thorough performance monitoring over multiple fine-tuning runs and several State-of-the-Art NLI datasets as well as contamination analysis. Our results highlight that generalization abilities are strongly correlated with model size. Furthermore, our findings underscore the importance of conducting multi-dataset assessments to correctly estimate OOD robustness.

## 2. Project Setup

Our code was developed on Linux using Python 3.11.10 and Cuda 11.8.0. Please make sure you have these specific versions installed on your system. On the ETH Euler cluster, you can load the following modules:

```bash
module load stack/2024-06 cuda/11.8.0 python_cuda
```

Create a Conda environment and install project dependencies by running:

```bash
conda create -n DL2024 python=3.11.10
conda activate DL2024
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
pip install -r requirements.txt
```

### Contamination checks
To run the contamination checks, you also need to setup BLEURT. To do so, follow the instructions below:

```bash
git clone https://github.com/google-research/bleurt.git dependencies/bleurt_scorer
cd dependencies/bleurt_scorer
pip install .
```

```bash
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
```

## 3. Repository Structure
The repository is structured as follows:
```
    .
    ├── configs/                   # TOML config files for training and contamination
    │   ├── mamba.toml 
    │   └── ...    
    ├── notebooks/                 # Jupyter notebooks for data analysis and visualization
    ├── results/                   # Folder containing the results of the experiments
    ├── models/                    # Default folder for model checkpoints
    ├── datasets/                  # Default folder for cached datasets
    ├── src/
    │   ├── data/                  # Data processing scripts
    │   │   └── NLIDataset.py
    │   ├── models/                # Model implementation scripts
    │   │   ├── MambaModel.py
    │   │   └── ...
    │   ├── contamination/         # Time-Travel and MinKPP scripts
    │   ├── train.py               # Main Python training script
    │   └── eval_contamination.py  # Main Python contamination analysis script
    ├── contamination.sh           # Slurm script for contamination analysis jobs
    ├── finetune.sh                # Slurm script for fine-tuning jobs
    └── requirements.txt
```

The scripts are meant to be run using SLURM. Modify either `finetune.sh` or `contamination.sh` as well as the relevant config files in the `configs/`, and submit the batch job.
Alternatively, you can run the scripts locally using `python src/train.py` or `python src/eval_contamination.py` and passing the argument `--config configs/<your_config_file>.toml`.


## 4. Final Notes

When using the ETH Euler cluster, the models and datasets need to first be cached on the local storage by running the python scripts, as the batch jobs are not able to access HuggingFace.

We use WandB for logging training metrics. If you want to use it, please create an account and set up the API key, otherwise modify `config.model.training.report_to` to "none".
