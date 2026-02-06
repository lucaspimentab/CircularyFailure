
#!/usr/bin/env python
"""Filter batch_*.parquet files to a standardized feature set.

Keeps id/time columns, drops label columns and *_imputed by default, and
optionally restricts to a keep-cols list.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _load_keep_cols(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"keep_cols_file not found: {p}")
    if p.suffix.lower() in {".csv", ".txt"}:
        cols = pd.read_csv(p, header=None).iloc[:, 0].dropna().astype(str).tolist()
    else:
        cols = p.read_text(encoding="utf-8").splitlines()
    return {c.strip() for c in cols if c.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter batch_*.parquet features.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--id_col", default="stay_id")
    parser.add_argument("--time_col", default="charttime")
    parser.add_argument("--keep_cols_file", default=None)
    parser.add_argument("--drop_suffixes", default="_imputed")
    parser.add_argument("--drop_cols", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = _load_keep_cols(args.keep_cols_file)
    drop_suffixes = [s for s in (args.drop_suffixes or "").split(",") if s]
    drop_cols = {c for c in (args.drop_cols or "").split(",") if c}

    files = sorted(input_dir.glob("batch_*.parquet"))
    files = [f for f in files if not f.name.endswith("_labels.parquet")]
    if not files:
        raise SystemExit(f"No batch_*.parquet found in {input_dir}")

    for i, path in enumerate(files, start=1):
        out_path = output_dir / path.name
        if out_path.exists() and not args.overwrite:
            continue

        cols = pq.read_schema(path).names
        keep = []
        for c in cols:
            if c in (args.id_col, args.time_col):
                keep.append(c)
                continue
            if keep_cols is not None and c not in keep_cols:
                continue
            if any(c.endswith(suf) for suf in drop_suffixes):
                continue
            if c in drop_cols:
                continue
            keep.append(c)

        df = pd.read_parquet(path, columns=keep)
        df.to_parquet(out_path, index=False)
        if i % 10 == 0 or i == 1 or i == len(files):
            print(f"[{i}/{len(files)}] {path.name} -> {out_path.name} ({len(keep)} cols)")

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
