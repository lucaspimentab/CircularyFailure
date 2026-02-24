# CircularyFailure

Temporal phenotyping and prediction of circulatory failure across MIMIC, eICU, and NWICU.

This repository provides code for preprocessing, feature engineering, MiniRocket clustering, and sequence modeling (GRU/LSTM/Transformer) with external validation workflows.

## Data governance (important)

- No raw clinical data should be committed.
- No patient-level outputs should be committed.
- Credentials/tokens must stay in `.env` (local only).
- `data/`, `outputs/`, `reports/`, and `runs/` are local-only directories.

## Repository structure

```text
configs/      # dataset configs
notebooks/    # preprocessing/modeling notebooks
scripts/      # CLI entry points
src/          # core code
data/         # local-only data (gitignored)
outputs/      # local-only artifacts (gitignored)
reports/      # local-only generated reports (gitignored)
```

## Quickstart

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Configure local env:

```bash
cp .env.example .env
```

Set in `.env`:

```text
DATA_DIR=...
OUTPUT_DIR=...
```

3) Run preprocessing (dataset-specific):

- use notebooks in `notebooks/preprocess/`, or
- run CLI wrappers in `scripts/preprocess/`.

4) Run clustering:

```bash
python scripts/clustering/run_minirocket_pipeline.py --dataset mimic
python scripts/clustering/run_minirocket_pipeline.py --dataset eicu
python scripts/clustering/run_minirocket_pipeline.py --dataset nwicu
```

5) Run evaluation:

```bash
python scripts/evaluation/run_evaluation.py --dataset mimic --k 4
```

## Reproducibility

- Configuration is centralized in `configs/*.yaml`.
- Pipelines are callable via `scripts/`.
- Keep random seeds fixed in configs for comparable experiments.

## License

To be defined by the authors before publication.
