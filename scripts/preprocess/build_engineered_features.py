#!/usr/bin/env python
"""Build per-stay engineered features as separate parquet files.

This script scans batch_*.parquet files and extracts columns that match
configured patterns (e.g., n_meas_, min_, max_, mean_, _instab, _intens, _cumul).
It aggregates to one row per stay_id (default: last available row by time).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _matches_patterns(col: str, patterns: list[str]) -> bool:
    return any(p in col for p in patterns)


def _select_engineered_cols(
    cols: list[str], patterns: list[str], drop_suffixes: list[str], drop_cols: set[str]
) -> list[str]:
    out: list[str] = []
    for c in cols:
        if c in drop_cols:
            continue
        if any(c.endswith(suf) for suf in drop_suffixes):
            continue
        if _matches_patterns(c, patterns):
            out.append(c)
    return out


def _ensure_time_sorted(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        return df.sort_values(id_col)

    s = df[time_col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return df.sort_values([id_col, time_col])

    # Try numeric offset
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        df = df.copy()
        df["__time_sort"] = s_num
        return df.sort_values([id_col, "__time_sort"]).drop(columns=["__time_sort"])

    # Fallback: parse datetime strings
    s_dt = pd.to_datetime(s, errors="coerce")
    if s_dt.notna().any():
        df = df.copy()
        df["__time_sort"] = s_dt
        return df.sort_values([id_col, "__time_sort"]).drop(columns=["__time_sort"])

    return df.sort_values(id_col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build engineered per-stay features from batch parquets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--id_col", default="stay_id")
    parser.add_argument("--time_col", default="charttime")
    parser.add_argument(
        "--patterns", default="n_meas_,min_,max_,mean_,_instab,_intens,_cumul"
    )
    parser.add_argument("--drop_suffixes", default="_imputed")
    parser.add_argument(
        "--drop_cols",
        default="stay_id,charttime,falencia,mortality,falencia_point,falencia_any",
    )
    parser.add_argument("--agg", choices=["last", "first"], default="last")
    parser.add_argument("--chunk_size", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_fh = open(args.log_path, "a", encoding="utf-8") if args.log_path else None

    def _log(msg: str) -> None:
        print(msg)
        if log_fh:
            log_fh.write(msg + "\n")

    existing = sorted(out_dir.glob("features_*.parquet"))
    if existing and not args.overwrite:
        raise SystemExit(f"{out_dir} already has engineered files. Use --overwrite to replace.")
    if existing and args.overwrite:
        for p in existing:
            p.unlink()

    files = sorted(glob.glob(str(input_dir / "batch_*.parquet")))
    files = [f for f in files if not f.endswith("_labels.parquet")]
    if not files:
        raise SystemExit(f"No feature parquets found in {input_dir} (excluding *_labels.parquet)")

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    drop_suffixes = [s.strip() for s in args.drop_suffixes.split(",") if s.strip()]
    drop_cols = {c.strip() for c in args.drop_cols.split(",") if c.strip()}

    buffer: list[pd.DataFrame] = []
    out_idx = 0
    seen: set = set()
    dup_count = 0

    for i, path in enumerate(files, start=1):
        _log(f"[{i}/{len(files)}] {Path(path).name}")
        file_cols = pq.read_schema(path).names
        eng_cols = _select_engineered_cols(file_cols, patterns, drop_suffixes, drop_cols)
        if not eng_cols:
            _log("  [WARN] No engineered columns matched in this file.")
            continue

        read_cols = [c for c in [args.id_col, args.time_col] if c in file_cols] + eng_cols
        df = pd.read_parquet(path, columns=read_cols)
        if args.id_col not in df.columns:
            _log("  [WARN] Missing id_col in this file, skipping.")
            continue

        df = _ensure_time_sorted(df, args.id_col, args.time_col)
        grouped = df.groupby(args.id_col, sort=False)
        if args.agg == "first":
            agg_df = grouped.first()
        else:
            agg_df = grouped.last()

        # track duplicates across files
        for sid in agg_df.index:
            if sid in seen:
                dup_count += 1
            seen.add(sid)

        agg_df = agg_df.reset_index()
        buffer.append(agg_df)

        if sum(len(b) for b in buffer) >= args.chunk_size:
            out_df = pd.concat(buffer, ignore_index=True).drop_duplicates(subset=[args.id_col])
            out_path = out_dir / f"features_{out_idx:04d}.parquet"
            out_df.to_parquet(out_path, index=False)
            _log(f"  saved {out_path} (rows={len(out_df)})")
            out_idx += 1
            buffer = []

    if buffer:
        out_df = pd.concat(buffer, ignore_index=True).drop_duplicates(subset=[args.id_col])
        out_path = out_dir / f"features_{out_idx:04d}.parquet"
        out_df.to_parquet(out_path, index=False)
        _log(f"  saved {out_path} (rows={len(out_df)})")

    if dup_count:
        _log(f"[WARN] {dup_count} stays appeared in more than one parquet; keeping last seen.")

    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
