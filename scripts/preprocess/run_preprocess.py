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


def _build_engineered(cfg, args, input_dir: Path, output_dir: Path) -> None:
    eng_cfg = cfg.get("features", {}).get("engineered", {})
    enabled = args.build_engineered or eng_cfg.get("enabled", False)
    if not enabled:
        return

    if not input_dir or not input_dir.exists():
        print(f"[WARN] engineered features: input_dir not found: {input_dir}")
        return
    files = sorted(input_dir.glob("batch_*.parquet"))
    files = [f for f in files if not f.name.endswith("_labels.parquet")]
    if not files:
        print(f"[WARN] engineered features: no batch_*.parquet found in {input_dir}")
        return

    eng_dir = Path(
        args.engineered_dir
        or eng_cfg.get("dir")
        or (output_dir / "preprocess" / "engineered_features")
    )
    patterns = args.engineered_patterns or ",".join(eng_cfg.get("patterns", []) or [])
    drop_cols = args.engineered_drop_cols or ",".join(eng_cfg.get("drop_cols", []) or [])
    drop_suffixes = args.engineered_drop_suffixes or ",".join(cfg.get("features", {}).get("drop_suffixes", []) or [])
    agg = args.engineered_agg or eng_cfg.get("agg", "last")
    chunk_size = args.engineered_chunk_size or eng_cfg.get("chunk_size", 200000)

    build_script = ROOT / "scripts" / "preprocess" / "build_engineered_features.py"
    cmd = [
        sys.executable,
        str(build_script),
        "--input_dir",
        str(input_dir),
        "--out_dir",
        str(eng_dir),
        "--id_col",
        cfg.get("columns", {}).get("id_col", "stay_id"),
        "--time_col",
        cfg.get("columns", {}).get("time_col", "charttime"),
        "--patterns",
        patterns,
        "--drop_suffixes",
        drop_suffixes,
        "--drop_cols",
        drop_cols,
        "--agg",
        agg,
        "--chunk_size",
        str(chunk_size),
    ]
    if args.engineered_overwrite:
        cmd += ["--overwrite"]
    run(cmd)



def _filter_batches(cfg, args, input_dir: Path, output_dir: Path) -> Path:
    keep_cols = cfg.get("features", {}).get("keep_cols_file")
    drop_suffixes = cfg.get("features", {}).get("drop_suffixes", [])
    drop_cols = cfg.get("features", {}).get("drop_cols", [])
    id_col = cfg.get("columns", {}).get("id_col", "stay_id")
    time_col = cfg.get("columns", {}).get("time_col", "charttime")

    filtered_dir = output_dir / "preprocess" / "batches" / "features_filtered"
    filter_script = ROOT / "scripts" / "preprocess" / "filter_feature_batches.py"
    cmd = [
        sys.executable,
        str(filter_script),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(filtered_dir),
        "--id_col",
        id_col,
        "--time_col",
        time_col,
        "--drop_suffixes",
        ",".join(drop_suffixes or []),
        "--drop_cols",
        ",".join(drop_cols or []),
    ]
    if keep_cols:
        cmd += ["--keep_cols_file", str(keep_cols)]
    run(cmd)
    return filtered_dir


def main():
    parser = argparse.ArgumentParser(description="Unified preprocess entrypoint")
    parser.add_argument("--dataset", required=True, choices=["mimic", "eicu", "nwicu"])
    parser.add_argument("--config_dir", default=str(Path("configs")))
    parser.add_argument(
        "--task",
        default="auto",
        choices=[
            "auto",
            "engineered_only",
            "eicu_h5",
            "eicu_windowed",
            "mimic_temporal",
            "mimic_h5",
            "nwicu_pipeline",
        ],
    )
    # Generic overrides
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--time_col", default=None)
    parser.add_argument("--step_min", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=168)
    parser.add_argument("--labels_path", default=None)
    parser.add_argument("--mortality_path", default=None)

    # Engineered features overrides
    parser.add_argument("--build_engineered", action="store_true")
    parser.add_argument("--engineered_patterns", default=None)
    parser.add_argument("--engineered_drop_cols", default=None)
    parser.add_argument("--engineered_drop_suffixes", default=None)
    parser.add_argument("--engineered_agg", default=None)
    parser.add_argument("--engineered_chunk_size", type=int, default=None)
    parser.add_argument("--engineered_overwrite", action="store_true")

    # eICU-specific
    parser.add_argument("--batch_dir", default=None)
    parser.add_argument("--engineered_dir", default=None)
    parser.add_argument("--split_tsv", default=None)
    parser.add_argument("--h5", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--duration_hours", type=float, default=None)
    parser.add_argument("--dt_hours", type=float, default=None)
    parser.add_argument("--grid_min", type=int, default=None)
    parser.add_argument("--label_mode", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)

    # NWICU-specific
    parser.add_argument("--grid_minutes", type=int, default=None)
    parser.add_argument("--fe_window", type=int, default=None)
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()

    cfg = load_config(Path(args.config_dir), args.dataset)

    # Task auto-selection
    if args.task == "auto":
        if args.dataset == "eicu":
            args.task = "eicu_h5"
        elif args.dataset == "mimic":
            args.task = "mimic_temporal"
        else:
            args.task = "nwicu_pipeline"


    if args.task == "engineered_only":
        input_dir = Path(args.input_dir or cfg.get("features_dir", ""))
        output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs"))
        _build_engineered(cfg, args, input_dir, output_dir)
        return

    if args.dataset == "mimic" and args.task == "mimic_temporal":
        input_dir = Path(args.input_dir or cfg.get("features_dir", ""))
        output_base = Path(args.output_dir or cfg.get("output_dir", "outputs"))
        if not input_dir.exists() or not list(input_dir.glob("batch_*.parquet")):
            raw_dir = output_base / "preprocess" / "batches" / "features_raw"
            if raw_dir.exists() and list(raw_dir.glob("batch_*.parquet")):
                input_dir = _filter_batches(cfg, args, raw_dir, output_base)
            else:
                print(f"[WARN] features input not found: {input_dir}")
        time_col = args.time_col or cfg.get("columns", {}).get("time_col", "charttime")
        output_dir = output_base / "clustering" / "minirocket"
        output_dir.mkdir(parents=True, exist_ok=True)

        _build_engineered(cfg, args, input_dir, output_base)

        build_script = ROOT / "src" / "clustering" / "minirocket" / "pipelines" / "build_temporal_dataset.py"
        cmd = [
            sys.executable,
            str(build_script),
            "--input_dir",
            str(input_dir),
            "--out_npy",
            str(output_dir / f"{args.dataset}_temporal.npy"),
            "--out_meta",
            str(output_dir / f"{args.dataset}_temporal_meta.parquet"),
            "--step_min",
            str(args.step_min),
            "--max_seq_len",
            str(args.max_seq_len),
            "--take",
            "first",
            "--time_col",
            time_col,
        ]
        if args.labels_path:
            cmd += ["--labels_path", str(Path(args.labels_path))]
        if args.mortality_path:
            cmd += ["--mortality_path", str(Path(args.mortality_path))]
        run(cmd)
        return


    if args.dataset == "mimic" and args.task == "mimic_h5":
        batch_dir = Path(args.batch_dir or cfg.get("features_dir", ""))
        output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs"))
        if not batch_dir.exists() or not list(batch_dir.glob("batch_*.parquet")):
            raw_dir = output_dir / "preprocess" / "batches" / "features_raw"
            if raw_dir.exists() and list(raw_dir.glob("batch_*.parquet")):
                batch_dir = _filter_batches(cfg, args, raw_dir, output_dir)
        h5_path = Path(args.h5 or cfg.get("etl", {}).get("h5", output_dir / "preprocess" / "h5" / "dataset_mimic.h5"))
        script = ROOT / "src" / "datasets" / "mimic" / "preprocess" / "build_h5_from_mimic.py"
        run([
            sys.executable,
            str(script),
            "--batch-dir",
            str(batch_dir),
            "--split-tsv",
            str(args.split_tsv or cfg.get("etl", {}).get("split_tsv", output_dir / "preprocess" / "split_all.tsv")),
            "--h5",
            str(h5_path),
            "--id-col",
            cfg.get("columns", {}).get("id_col", "stay_id"),
            "--time-col",
            cfg.get("columns", {}).get("time_col", "charttime"),
            "--label-col",
            cfg.get("columns", {}).get("falencia_col", "falencia"),
            "--seed",
            str(args.seed),
            "--train-frac",
            str(args.train_frac),
            "--val-frac",
            str(args.val_frac),
        ])
        return

    if args.dataset == "nwicu" and args.task == "nwicu_pipeline":
        input_dir = Path(args.input_dir or cfg.get("input_dir", ""))
        output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs"))

        script = ROOT / "src" / "datasets" / "nwicu" / "scripts" / "run_pipeline.py"
        grid_minutes = args.grid_minutes or cfg.get("nwicu", {}).get("grid_minutes", 5)
        fe_window = args.fe_window or cfg.get("nwicu", {}).get("fe_window", 12)

        # 1) merge
        run([
            sys.executable,
            str(script),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-level",
            args.log_level,
            "prep_merge",
        ])
        # 2) impute
        run([
            sys.executable,
            str(script),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-level",
            args.log_level,
            "impute",
            "--grid-minutes",
            str(grid_minutes),
        ])
        # 3) labels
        run([
            sys.executable,
            str(script),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-level",
            args.log_level,
            "label",
        ])
        # 4) feature extraction
        run([
            sys.executable,
            str(script),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-level",
            args.log_level,
            "fe_extract",
            "--window",
            str(fe_window),
        ])
        # 4b) filter features -> features_filtered
        raw_dir = output_dir / "preprocess" / "batches" / "features_raw"
        if raw_dir.exists() and list(raw_dir.glob("batch_*.parquet")):
            _filter_batches(cfg, args, raw_dir, output_dir)
        # 5) build H5 features
        cmd = [
            sys.executable,
            str(script),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-level",
            args.log_level,
            "build_features",
            "--features-dir",
            str(output_dir / "preprocess" / "batches" / "features_filtered"),
            "--out",
            str(output_dir / "preprocess" / "h5" / "dataset_features.h5"),
        ]
        if args.split_path:
            cmd += ["--split-path", str(Path(args.split_path))]
        run(cmd)

        _build_engineered(cfg, args, Path(args.input_dir or cfg.get("features_dir", "")), output_dir)
        return

    if args.dataset != "eicu":
        raise SystemExit(f"Unsupported task {args.task} for dataset {args.dataset}.")

    output_dir = Path(cfg.get("output_dir", "outputs"))
    features_dir = Path(cfg.get("features_dir", ""))

    batch_dir = Path(args.batch_dir or cfg.get("etl", {}).get("batch_dir", features_dir))
    if not batch_dir.exists() or not list(batch_dir.glob("batch_*.parquet")):
        raw_dir = output_dir / "preprocess" / "batches" / "features_raw"
        if raw_dir.exists() and list(raw_dir.glob("batch_*.parquet")):
            batch_dir = _filter_batches(cfg, args, raw_dir, output_dir)
    engineered_dir = Path(args.engineered_dir or cfg.get("etl", {}).get("engineered_dir", output_dir / "preprocess" / "engineered_features"))
    split_tsv = Path(args.split_tsv or cfg.get("etl", {}).get("split_tsv", output_dir / "preprocess" / "split_all.tsv"))
    h5 = Path(args.h5 or cfg.get("etl", {}).get("h5", output_dir / "preprocess" / "h5" / "dataset_eicu.h5"))
    label_col = args.label_col or cfg.get("etl", {}).get("label_col", "falencia")

    if args.task == "eicu_h5":
        script = ROOT / "src" / "datasets" / "eicu" / "preprocess" / "build_h5_from_eicu.py"
        run([
            sys.executable,
            str(script),
            "--batch-dir",
            str(batch_dir),
            "--features-dir",
            str(engineered_dir),
            "--split-tsv",
            str(split_tsv),
            "--h5",
            str(h5),
            "--label-col",
            str(label_col),
            "--seed",
            str(args.seed),
            "--train-frac",
            str(args.train_frac),
            "--val-frac",
            str(args.val_frac),
        ])
    elif args.task == "eicu_windowed":
        script = ROOT / "src" / "datasets" / "eicu" / "preprocess" / "build_h5_windowed_eicu.py"
        windowed_h5 = Path(cfg.get("etl", {}).get("windowed_h5", output_dir / "preprocess" / "h5" / "dataset_eicu_windowed.h5"))
        duration_hours = args.duration_hours or cfg.get("etl", {}).get("duration_hours", 8.0)
        dt_hours = args.dt_hours or cfg.get("etl", {}).get("dt_hours", 0.1)
        grid_min = args.grid_min or cfg.get("etl", {}).get("grid_min", 5)
        label_mode = args.label_mode or cfg.get("etl", {}).get("label_mode", "window")
        chunk_size = args.chunk_size or cfg.get("etl", {}).get("chunk_size", 200000)
        run([
            sys.executable,
            str(script),
            "--input-h5",
            str(h5),
            "--output-h5",
            str(windowed_h5),
            "--duration-hours",
            str(duration_hours),
            "--dt-hours",
            str(dt_hours),
            "--grid-min",
            str(grid_min),
            "--label-mode",
            str(label_mode),
            "--seed",
            str(args.seed),
            "--chunk-size",
            str(chunk_size),
        ])
    else:
        raise SystemExit(f"Unsupported task {args.task} for dataset eicu.")


if __name__ == "__main__":
    main()
