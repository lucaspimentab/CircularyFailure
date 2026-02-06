#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Cluster outcomes summary (mortality/falencia rates)")
    parser.add_argument("--dataset", required=True, choices=["mimic", "eicu", "nwicu"])
    parser.add_argument("--config_dir", default=str(Path("configs")))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--cluster_embeddings", default=None)
    parser.add_argument("--falencia_summary", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--id_col", default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config_dir), args.dataset)
    output_base = Path(cfg.get("output_dir", "outputs"))
    id_col = args.id_col or cfg.get("columns", {}).get("id_col", "stay_id")

    cluster_embeddings = Path(
        args.cluster_embeddings
        or output_base / "clustering" / "minirocket" / "clusters_minirocket" / f"cluster_embeddings_kmeans_k{args.k}.parquet"
    )
    falencia_summary = Path(
        args.falencia_summary
        or output_base / "preprocess" / "falencia_stay_summary.csv"
    )
    out_csv = Path(
        args.out_csv
        or output_base / "clustering" / "minirocket" / "clusters_minirocket" / f"cluster_outcomes_k{args.k}_allfal.csv"
    )

    if not cluster_embeddings.exists():
        raise SystemExit(f"Missing cluster embeddings: {cluster_embeddings}")
    if not falencia_summary.exists():
        raise SystemExit(f"Missing falencia summary: {falencia_summary}")

    emb = pd.read_parquet(cluster_embeddings)
    out = pd.read_csv(falencia_summary)

    df = emb.merge(out, on=id_col, how="left")
    res = df.groupby("cluster", as_index=False).agg(
        n=(id_col, "count"),
        falencia_normal_rate=("falencia_normal", "mean"),
        falencia_45_rate=("falencia_2of3_45min", "mean"),
        falencia_60_rate=("falencia_2of3_60min", "mean"),
        mortality_rate=("mortality", "mean"),
    )
    res["dataset"] = args.dataset.upper()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
