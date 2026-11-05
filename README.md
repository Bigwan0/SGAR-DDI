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
```text
Update README installation instructions
## Reproducing the Main Results and Figures

The main experimental results can be reproduced using the training and evaluation procedures described in this repository.

### Experimental Results

The DrugBank transductive and inductive experiments and the TWOSIDES transductive experiments can be run using the corresponding commands provided above. The reported performance metrics are aggregated across three repetitions or predefined folds.

### Tables and Figures

The performance tables are compiled from the evaluation metrics obtained for each repetition or fold. The ablation results are obtained by evaluating the corresponding model variants under the same experimental settings.

ROC and precision–recall curves can be generated from the ground-truth labels and prediction scores obtained during evaluation. Other performance figures can be reconstructed from the corresponding evaluation metrics.

The original plotting scripts and saved prediction outputs are not included in this repository.

### Reproducibility Settings

The environment configuration is provided in `environment.yml`. The experimental hyperparameters, validation procedure, and checkpoint-selection criterion are described in the manuscript and Supporting Information.
