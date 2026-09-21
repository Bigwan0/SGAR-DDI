# SGAR-DDI

## Requirements

- Python 3.10
- PyTorch 2.0.1
- CUDA 11.8
- PyTorch Geometric 2.3.1
- RDKit
- NumPy
- Pandas
- scikit-learn

## Installation

Create the environment with Conda:

```bash
conda env create -f environment.yml
conda activate sgar_ddi
```

Alternatively:

```bash
bash Install
conda activate sgar_ddi
```

## Dataset

The processed datasets are based on the DSN-DDI repository:

https://github.com/microsoft/Drug-Interaction-Research/blob/DSN-DDI-for-DDI-Prediction/DSN-DDI-dataset.zip

Download and extract the dataset before running the experiments.

## Run

DrugBank transductive:

```bash
python "drugbank_test - transductive/transductive_train.py" --repeat 0 --seed 0
```

DrugBank inductive:

```bash
python "drugbank_test - inductive/inductive_train.py" --fold 0 --seed 0
```

TWOSIDES:

```bash
python "twosides_test/train_script.py" --repeat 0 --seed 0
```

Use repeat/fold values `0`, `1`, and `2` for the three experimental runs.
