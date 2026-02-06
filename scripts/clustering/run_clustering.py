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
    parser = argparse.ArgumentParser(description="Unified clustering entrypoint")
    parser.add_argument("--dataset", required=True, choices=["mimic", "eicu", "nwicu"])
    parser.add_argument("--config_dir", default=str(Path("configs")))
    parser.add_argument("--method", default="minirocket", choices=["minirocket"])
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config_dir), args.dataset)
    output_base = Path(cfg.get("output_dir", "outputs"))

    if args.method != "minirocket":
        raise SystemExit("Only minirocket is implemented in the unified CLI.")

    output_dir = Path(args.output_dir or (output_base / "clustering" / "minirocket" / "clusters_minirocket"))
    emb = Path(args.embeddings or (output_base / "clustering" / "minirocket" / f"{args.dataset}_minirocket_emb.npy"))
    meta = Path(args.meta or (output_base / "clustering" / "minirocket" / f"{args.dataset}_minirocket_meta.parquet"))

    k_min = args.k_min or cfg.get("minirocket", {}).get("k_min", 2)
    k_max = args.k_max or cfg.get("minirocket", {}).get("k_max", 6)


    use_engineered = cfg.get("minirocket", {}).get("use_engineered", False) or cfg.get("features", {}).get("engineered", {}).get("enabled", False)
    engineered_dir = cfg.get("features", {}).get("engineered", {}).get("dir")

    script = ROOT / "src" / "clustering" / "minirocket" / "pipelines" / "cluster_minirocket.py"
    cmd = [
        sys.executable,
        str(script),
        "--embeddings",
        str(emb),
        "--meta",
        str(meta),
        "--output_dir",
        str(output_dir),
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
