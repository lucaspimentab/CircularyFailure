# CircularyFailure

**Temporal phenotyping and prediction of circulatory failure across MIMIC, eICU, and NWICU.**  
This repository provides a reproducible pipeline to preprocess ICU time?series, build engineered features, generate MiniRocket embeddings, cluster patient trajectories, and train sequence models (GRU/LSTM/Transformer) for outcome prediction and external validation.

---

## What this repo delivers

- **Standardized preprocessing** for MIMIC, eICU, and NWICU
- **Temporal clustering** with MiniRocket + K?Means (K=2..6)
- **Phenotype summaries** (means + quantiles) per cluster
- **Outcomes per cluster** (mortality + circulatory failure: normal / 45min / 60min)
- **Sequence models** (GRU/LSTM/Transformer) trained on HDF5 datasets
- **Cross?dataset comparison** and reports

---

## Datasets

This project expects access to the ICU datasets below. **Raw data are NOT stored in the repo.**

- **MIMIC** (time?series with labs, vitals, engineered features)
- **eICU** (time?series with engineered features)
- **NWICU (CircEWS)**

You configure data locations via `.env` + `configs/*.yaml`.

---

## Repository structure

```
configs/                # dataset configs (mimic/eicu/nwicu)
notebooks/
  preprocess/           # dataset?specific preprocessing (queries + export)
  modeling/             # sequence model notebook
scripts/                # CLI entry points
src/
  common/               # shared utils + MiniRocket utilities
  clustering/minirocket/
    core/               # MiniRocket implementation
    pipelines/          # build/embed/cluster/summarize/falencia
outputs/                # generated artifacts (gitignored)
reports/                # final tables + narrative report
```

---

## Quickstart

### 1) Environment
```bash
pip install -r requirements.txt
```

### 2) Configure `.env`
Copy `.env.example` ? `.env` and set:
```
DATA_DIR=...
OUTPUT_DIR=...
```

### 3) Preprocess (per dataset)
Run the dataset?specific notebook in `notebooks/preprocess/` **from top to bottom**.
Each notebook exports:
- batches/parquets
- engineered features
- HDF5 datasets
- `falencia_stay_summary.csv` (normal/45/60 + mortality)

### 4) MiniRocket clustering
```bash
python scripts/clustering/run_minirocket_pipeline.py --dataset mimic
python scripts/clustering/run_minirocket_pipeline.py --dataset eicu
python scripts/clustering/run_minirocket_pipeline.py --dataset nwicu
```

### 5) Sequence models (optional)
```bash
python scripts/modeling/train_sequence_model.py --dataset mimic --model gru
```

### 6) Evaluation (cluster outcomes)
```bash
python scripts/evaluation/run_evaluation.py --dataset mimic --k 4
```

---

## Output layout (per dataset)

```
outputs/<dataset>/
  preprocess/
    batches/
    engineered_features/
    h5/
    falencia_stay_summary.csv
    mortality_by_stay.csv
  clustering/minirocket/
    <dataset>_temporal.npy
    <dataset>_minirocket_emb.npy
    clusters_minirocket/
  models/
  reports/
```

---

## Reports

- `reports/report_full_mimic_eicu_nwicu.txt` ? full narrative report
- `reports/phenotype_k4_report.csv` ? phenotype summary (K=4)
- `reports/compare_k4_summary.csv` ? cross?dataset comparison (K=4)

---

## Reproducibility

All key steps have CLI wrappers in `scripts/` and are driven by configs in `configs/`.
Seeds are fixed by default. Logs and metrics are stored in `outputs/`.