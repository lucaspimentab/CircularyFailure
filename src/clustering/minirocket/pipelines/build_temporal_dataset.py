from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.minirocket_utils import detect_id_col, detect_time_col, select_feature_columns, setup_logger, read_table


def _read_schema_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    except Exception:
        df = pd.read_parquet(path)
        return list(df.columns)


def _sample_ids(ids: np.ndarray, max_stays: int, seed: int) -> np.ndarray:
    if max_stays is None or max_stays >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    return rng.choice(ids, size=max_stays, replace=False)


def _compute_mean_std(memmap: np.ndarray, chunk: int = 256):
    n, _, d = memmap.shape
    sum_v = np.zeros(d, dtype=np.float64)
    cnt_v = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        block = memmap[i : i + chunk]
        mask = ~np.isnan(block)
        sum_v += np.nansum(block, axis=(0, 1))
        cnt_v += mask.sum(axis=(0, 1))
    mean = np.divide(sum_v, cnt_v, out=np.zeros_like(sum_v), where=cnt_v > 0)

    var_sum = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        block = memmap[i : i + chunk]
        diff = block - mean
        diff[np.isnan(diff)] = 0.0
        var_sum += np.sum(diff * diff, axis=(0, 1))
    var = np.divide(var_sum, cnt_v, out=np.zeros_like(var_sum), where=cnt_v > 0)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize_inplace(memmap: np.ndarray, mean: np.ndarray, std: np.ndarray, chunk: int = 256):
    n = memmap.shape[0]
    for i in range(0, n, chunk):
        block = memmap[i : i + chunk]
        block = (block - mean) / std
        memmap[i : i + chunk] = block


def _add_time_bins(df: pd.DataFrame, id_col: str, time_col: str, step_min: int) -> pd.DataFrame:
    """Add __bin column based on time_col (numeric minutes or datetime)."""
    series = df[time_col]
    if isinstance(series, pd.DataFrame):
        # Handle duplicated column names by taking the first one.
        series = series.iloc[:, 0]
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    if not is_datetime and series.dtype == object:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().any():
            series = parsed
            is_datetime = True

    if is_datetime:
        df = df.dropna(subset=[time_col]).copy()
        df["_t"] = pd.to_datetime(series, errors="coerce")
        df = df.dropna(subset=["_t"])
        min_t = df.groupby(id_col)["_t"].transform("min")
        df["__offset_min"] = (df["_t"] - min_t).dt.total_seconds() / 60.0
        df["__bin"] = (df["__offset_min"] // step_min).astype(int)
        return df.drop(columns=["_t", "__offset_min"])

    df = df.dropna(subset=[time_col]).copy()
    df["__bin"] = (df[time_col].astype(float) // step_min).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal dataset from parquet batches.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_npy", type=str, required=True)
    parser.add_argument("--out_meta", type=str, required=True)
    parser.add_argument("--labels_path", type=str, default=None)
    parser.add_argument("--mortality_path", type=str, default=None)
    parser.add_argument("--stays_list", type=str, default=None)
    parser.add_argument("--max_stays", type=int, default=None)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--step_min", type=int, default=60)
    parser.add_argument("--max_seq_len", type=int, default=96)
    parser.add_argument("--take", type=str, default="first", choices=["first", "last"])
    parser.add_argument("--time_col", type=str, default=None, help="Override time column name.")
    parser.add_argument("--keep_imputed", action="store_true", default=False)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--scaler_out", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger(Path(args.log_path) if args.log_path else None)

    input_dir = Path(args.input_dir)
    paths = sorted(
        p
        for p in input_dir.glob("batch_*.parquet")
        if not p.name.endswith("_labels.parquet")
    )
    if not paths:
        raise SystemExit(f"No batch_*.parquet found in {input_dir}")

    columns = _read_schema_columns(paths[0])
    id_col = detect_id_col(columns)
    if args.time_col:
        time_col = args.time_col
        if time_col not in columns:
            raise SystemExit(f"time_col '{time_col}' not found in input files.")
    else:
        try:
            time_col = detect_time_col(columns)
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
    feat_cols = select_feature_columns(columns, drop_imputed=not args.keep_imputed)
    # Safety: remove id/time if they slipped into feature list
    feat_cols = [c for c in feat_cols if c not in {id_col, time_col}]
    if not feat_cols:
        raise SystemExit("No feature columns found after filtering.")
    logger.info("Using time column: %s", time_col)

    labels_df = None
    mortality_df = None
    if args.labels_path:
        labels_df = read_table(Path(args.labels_path))
        if id_col not in labels_df.columns:
            if "patientunitstayid" in labels_df.columns:
                labels_df = labels_df.rename(columns={"patientunitstayid": id_col})
            elif "stay_id" in labels_df.columns:
                labels_df = labels_df.rename(columns={"stay_id": id_col})
    if args.mortality_path:
        mortality_df = read_table(Path(args.mortality_path))
        if id_col not in mortality_df.columns:
            if "patientunitstayid" in mortality_df.columns:
                mortality_df = mortality_df.rename(columns={"patientunitstayid": id_col})
            elif "stay_id" in mortality_df.columns:
                mortality_df = mortality_df.rename(columns={"stay_id": id_col})

    if args.stays_list:
        stays = read_table(Path(args.stays_list))[id_col].astype(int).unique()
    elif labels_df is not None:
        stays = labels_df[id_col].astype(int).unique()
    else:
        logger.info("No labels/stays list provided; scanning parquets for stay ids.")
        stay_set = set()
        for path in paths:
            df_ids = pd.read_parquet(path, columns=[id_col])
            stay_set.update(df_ids[id_col].dropna().astype(int).unique().tolist())
        stays = np.array(sorted(stay_set), dtype=int)

    stays = _sample_ids(stays, args.max_stays, args.seed)
    stays = np.array(sorted(stays), dtype=int)
    n_stays = len(stays)
    logger.info("Using %d stays", n_stays)

    id_to_idx = {int(s): i for i, s in enumerate(stays)}
    stay_set = set(id_to_idx.keys())

    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    data = np.lib.format.open_memmap(
        out_npy, mode="w+", dtype="float32", shape=(n_stays, args.max_seq_len, len(feat_cols))
    )
    data[:] = np.nan
    lengths = np.zeros(n_stays, dtype=np.int32)

    seen = set()
    for i, path in enumerate(tqdm(paths, desc="parquets", unit="file"), start=1):
        logger.info("Processing %s (%d/%d)", path.name, i, len(paths))
        df = pd.read_parquet(path, columns=[id_col, time_col] + feat_cols)
        if df.columns.duplicated().any():
            # Some MIMIC parquets can contain duplicate column names (e.g., charttime)
            df = df.loc[:, ~df.columns.duplicated()]
        df = df[df[id_col].isin(stay_set)]
        if df.empty:
            continue
        df = df.dropna(subset=[id_col, time_col])
        df[id_col] = df[id_col].astype(int)
        df = _add_time_bins(df, id_col, time_col, args.step_min)
        agg = df.groupby([id_col, "__bin"], sort=True)[feat_cols].mean().reset_index()
        for sid, g in agg.groupby(id_col):
            idx = id_to_idx.get(int(sid))
            if idx is None:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            g = g.sort_values("__bin")
            vals = g[feat_cols].to_numpy(dtype=np.float32)
            if args.take == "last":
                vals = vals[-args.max_seq_len :]
            else:
                vals = vals[: args.max_seq_len]
            data[idx, : len(vals), :] = vals
            lengths[idx] = len(vals)

    logger.info("Filled %d/%d stays", len(seen), n_stays)

    meta = pd.DataFrame({id_col: stays, "length": lengths})
    if labels_df is not None:
        meta = meta.merge(labels_df, on=id_col, how="left")
    if mortality_df is not None:
        meta = meta.merge(mortality_df, on=id_col, how="left")
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    if out_meta.suffix.lower() == ".parquet":
        meta.to_parquet(out_meta, index=False)
    else:
        meta.to_csv(out_meta, index=False)
    logger.info("Meta saved to %s", out_meta)

    if args.standardize:
        logger.info("Computing mean/std for standardization...")
        mean, std = _compute_mean_std(data)
        _standardize_inplace(data, mean, std)
        scaler_out = Path(args.scaler_out) if args.scaler_out else out_npy.with_suffix(".scaler.npz")
        np.savez(scaler_out, mean=mean, std=std)
        logger.info("Standardized data saved. Scaler: %s", scaler_out)


if __name__ == "__main__":
    main()