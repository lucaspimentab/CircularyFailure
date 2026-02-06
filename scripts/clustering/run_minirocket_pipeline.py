#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Unified MiniRocket pipeline")
    parser.add_argument("--dataset", required=True, choices=["mimic", "eicu", "nwicu"])
    parser.add_argument("--config_dir", default=str(Path("configs")))
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_embed", action="store_true")
    parser.add_argument("--skip_cluster", action="store_true")
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--step_min", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=168)
    parser.add_argument("--time_col", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config_dir), args.dataset)

    input_dir = Path(args.input_dir or cfg.get("features_dir", ""))
    if not input_dir:
        raise SystemExit("features_dir not set; use --input_dir or set in config")

    output_base = Path(args.output_dir or cfg.get("output_dir", ""))
    if not output_base:
        raise SystemExit("output_dir not set; use --output_dir or set in config")
    output_dir = output_base / "clustering" / "minirocket"
    output_dir.mkdir(parents=True, exist_ok=True)


    use_engineered = cfg.get("minirocket", {}).get("use_engineered", False) or cfg.get("features", {}).get("engineered", {}).get("enabled", False)
    engineered_dir = cfg.get("features", {}).get("engineered", {}).get("dir")

    time_col = args.time_col or cfg.get("columns", {}).get("time_col", "charttime")

    build_script = ROOT / "src" / "clustering" / "minirocket" / "pipelines" / "build_temporal_dataset.py"
    embed_script = ROOT / "src" / "clustering" / "minirocket" / "pipelines" / "embed_minirocket.py"
    cluster_script = ROOT / "src" / "clustering" / "minirocket" / "pipelines" / "cluster_minirocket.py"

    out_npy = output_dir / f"{args.dataset}_temporal.npy"
    out_meta = output_dir / f"{args.dataset}_temporal_meta.parquet"
    emb_out = output_dir / f"{args.dataset}_minirocket_emb.npy"
    emb_meta = output_dir / f"{args.dataset}_minirocket_meta.parquet"
    cluster_out = output_dir / "clusters_minirocket"

    if not args.skip_build:
        run([
            sys.executable,
            str(build_script),
            "--input_dir",
            str(input_dir),
            "--out_npy",
            str(out_npy),
            "--out_meta",
            str(out_meta),
            "--step_min",
            str(args.step_min),
            "--max_seq_len",
            str(args.max_seq_len),
            "--take",
            "first",
            "--time_col",
            time_col,
        ])

    if not args.skip_embed:
        run([
            sys.executable,
            str(embed_script),
            "--dataset",
            str(out_npy),
            "--meta_in",
            str(out_meta),
            "--emb_out",
            str(emb_out),
            "--meta_out",
            str(emb_meta),
            "--n_kernels",
            str(cfg.get("minirocket", {}).get("n_kernels", 1024)),
            "--fit_stays",
            str(cfg.get("minirocket", {}).get("fit_stays", 50000)),
            "--batch_size",
            str(cfg.get("minirocket", {}).get("batch_size", 128)),
        ])

    if not args.skip_cluster:
        k_min = args.k_min or cfg.get("minirocket", {}).get("k_min", 2)
        k_max = args.k_max or cfg.get("minirocket", {}).get("k_max", 6)
        cmd = [
            sys.executable,
            str(cluster_script),
            "--embeddings",
            str(emb_out),
            "--meta",
            str(emb_meta),
            "--output_dir",
            str(cluster_out),
            "--clusterer",
            "kmeans",
            "--k_min",
            str(k_min),
            "--k_max",
            str(k_max),
            "--kmeans_epochs",
            str(cfg.get("minirocket", {}).get("kmeans_epochs", 10)),
            "--kmeans_batch",
            str(cfg.get("minirocket", {}).get("kmeans_batch", 4096)),
            "--pca_dim",
            str(cfg.get("minirocket", {}).get("pca_dim", 50)),
            "--pca_sample",
            str(cfg.get("minirocket", {}).get("pca_sample", 100000)),
        ]
        if use_engineered and engineered_dir:
            cmd += ["--use_engineered", "--engineered_dir", str(engineered_dir)]
        run(cmd)


if __name__ == "__main__":
    main()
