# SGAR-DDI

## Requirement

To run the code, the following dependencies are required:

- Python == 3.7.12
- PyTorch == 1.9.0
- CUDA Toolkit == 11.1
- PyTorch Geometric == 2.0.3
- RDKit == 2020.09.1

## Installation

You can create a virtual environment using Conda:

```bash
bash Install
```

Then activate the environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sgar_ddi
```

Alternatively, you can create the environment using `environment.yml`:

```bash
conda env create -f environment.yml
conda activate sgar_ddi
```

## Dataset

The processed datasets used in this study were obtained from the DSN-DDI repository:

https://github.com/microsoft/Drug-Interaction-Research/blob/DSN-DDI-for-DDI-Prediction/DSN-DDI-dataset.zip

After downloading and extracting the dataset, place the following data folders in the root directory of SGAR-DDI:

```text
SGAR-DDI/
├── drugbank/
├── inductive_data/
└── twosides/
```

## Run

Run the DrugBank inductive experiment:

```bash
python "drugbank_test - inductive/inductive_train.py"
```

Run the DrugBank transductive experiment:

```bash
python "drugbank_test - transductive/transductive_train.py"
```

Run the TWOSIDES experiment:

```bash
python "twosides_test/train.py"
```
