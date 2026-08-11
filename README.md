# Predicting Circulatory Failure in ICU Patients with Deep Sequential Models

This repository archives the implementation associated with the paper
*Predicting Circulatory Failure in ICU Patients with Deep Sequential Models*,
presented at CBIC 2025.

- Event page: https://sbia.org.br/eventos/cbic_2025/cbic2025-1191697/
- DOI: https://doi.org/10.21528/CBIC2025-1191697
- Dataset: MIMIC-IV
- Prediction horizon: up to 8 hours before circulatory failure
- Temporal resolution: 5 minutes

The expanded multi-cohort work for the journal manuscript is maintained in
[`lucaspimentab/multicohort-circulatory-failure`](https://github.com/lucaspimentab/multicohort-circulatory-failure).

## Models

The study compares three deep sequential architectures:

- Transformer
- LSTM
- GRU

Pretrained weights for three random seeds per architecture are included under
`models/`.

## Repository Structure

```text
data/
  features_names.csv       # model feature names
  metrics.csv              # aggregate evaluation metrics
figures/                   # publication figures
models/                    # pretrained model weights
notebooks/
  extract_stayids_and_split.ipynb
  preprocess.ipynb
  models.ipynb
  graphs.ipynb
environment.yml
requirements.txt
```

Patient identifiers, raw MIMIC-IV data, and generated patient-level datasets
are not distributed in this repository.

## Setup

Using Conda:

```bash
conda env create -f environment.yml
conda activate circulatory-failure
```

Or using pip:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure the local MIMIC-IV PostgreSQL
connection. Credentials must never be committed.

## Workflow

1. Run `notebooks/extract_stayids_and_split.ipynb` to create patient-level
   train, validation, and test splits locally.
2. Run `notebooks/preprocess.ipynb` to build the model-ready HDF5 dataset.
3. Run `notebooks/models.ipynb` to train and evaluate the sequence models.
4. Run `notebooks/graphs.ipynb` to generate the evaluation figures.

## Reported Results

| Model | AUROC | AUPRC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 0.976 | 0.884 | 0.804 | 0.844 | 0.768 |
| LSTM | 0.969 | 0.858 | 0.745 | 0.839 | 0.671 |
| GRU | 0.969 | 0.839 | 0.782 | 0.812 | 0.755 |

## Data Governance

MIMIC-IV is not distributed with this repository. Access requires PhysioNet
credentialing and compliance with the applicable data use agreement. Keep all
credentials, raw data, patient identifiers, and patient-level outputs outside
Git.
