# Scripts (CLI entry points)

This folder contains **command-line entry points** that call the core logic in `src/`.
Use these to run pipelines without touching notebooks.

---

## Preprocess

### `scripts/preprocess/run_preprocess.py`
Runs dataset preprocessing based on `configs/<dataset>.yaml`.
Generates batches/parquets, engineered features, HDF5, and falencia+mortality summary.

**Example:**
```bash
python scripts/preprocess/run_preprocess.py --dataset mimic
python scripts/preprocess/run_preprocess.py --dataset eicu
python scripts/preprocess/run_preprocess.py --dataset nwicu
```

---

## Clustering (MiniRocket)

### `scripts/clustering/run_minirocket_pipeline.py`
Full MiniRocket pipeline: build temporal dataset ? embeddings ? k?means clustering.

**Example:**
```bash
python scripts/clustering/run_minirocket_pipeline.py --dataset mimic
```

### `scripts/clustering/run_clustering.py`
Runs clustering only (if embeddings already exist).

**Example:**
```bash
python scripts/clustering/run_clustering.py --dataset mimic
```

---

## Modeling

### `scripts/modeling/train_sequence_model.py`
Trains GRU/LSTM/Transformer from HDF5 datasets.
Saves models + metrics into `outputs/<dataset>/models/`.

**Example:**
```bash
python scripts/modeling/train_sequence_model.py --dataset mimic --model gru
```

---

## Evaluation

### `scripts/evaluation/run_evaluation.py`
Generates outcome summaries by cluster (mortality + fal?ncia normal/45/60).

**Example:**
```bash
python scripts/evaluation/run_evaluation.py --dataset mimic --k 4
```

---

## Notes

- All scripts read dataset-specific settings from `configs/*.yaml`.
- Configure paths via `.env` (see `.env.example`).
- Run from the repository root.
