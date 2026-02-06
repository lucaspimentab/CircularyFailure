import argparse
import glob
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _rolling_mean_ignore_nan(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean ignoring NaNs. Returns array of same length with NaNs at edges."""
    if window <= 1:
        return values.copy()
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    # No full window fits: keep all-NaN result (same behavior as strict centered rolling window).
    if n < window:
        return out
    valid = ~np.isnan(values)
    if not valid.any():
        return out
    vals = np.nan_to_num(values, nan=0.0)
    kernel = np.ones(window, dtype=np.float32)
    sum_w = np.convolve(vals, kernel, mode="valid")
    cnt_w = np.convolve(valid.astype(np.float32), kernel, mode="valid")
    mean_w = np.divide(sum_w, cnt_w, out=np.full_like(sum_w, np.nan), where=cnt_w > 0)
    half = window // 2
    out[half : half + mean_w.shape[0]] = mean_w
    return out


def _compute_event_any(
    df: pd.DataFrame,
    time_col: str,
    mbp_col: str,
    vaso_col: str,
    lactate_col: str,
    step_min: int,
    window_min: int,
) -> bool:
    if df.empty:
        return False
    df = _resample_to_step(df, time_col, step_min)
    if df.empty:
        return False

    mbp = df[mbp_col] if mbp_col in df.columns else pd.Series(index=df.index, dtype=float)
    vaso = df[vaso_col] if vaso_col in df.columns else pd.Series(index=df.index, dtype=float)
    lact = df[lactate_col] if lactate_col in df.columns else pd.Series(index=df.index, dtype=float)

    map_or_drugs = (mbp <= 65) | (vaso > 0)
    # If both mbp and vaso are missing, mark as NaN (unknown)
    missing_md = mbp.isna() & vaso.isna()
    map_or_drugs = map_or_drugs.astype("float32")
    map_or_drugs[missing_md] = np.nan

    lact_above = (lact >= 2).astype("float32")
    lact_above[lact.isna()] = np.nan

    window = int(round(window_min / step_min))
    if window < 1:
        window = 1

    map_mean = _rolling_mean_ignore_nan(map_or_drugs.to_numpy(), window)
    lact_mean = _rolling_mean_ignore_nan(lact_above.to_numpy(), window)

    event = (map_mean >= (2.0 / 3.0)) & (lact_mean >= (2.0 / 3.0))
    return bool(np.nanmax(event) if event.size else False)


def _resample_to_step(df: pd.DataFrame, time_col: str, step_min: int) -> pd.DataFrame:
    """Resample using datetime index or numeric offset-minutes, whichever is available."""
    if time_col not in df.columns:
        return df.iloc[0:0]

    s = df[time_col]
    # Prefer true datetime if it is already datetime-like.
    if pd.api.types.is_datetime64_any_dtype(s):
        tmp = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
        return tmp.resample(f"{step_min}min").median()

    # Try numeric offset (e.g., eICU offset_min).
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() > 0:
        tmp = df.loc[s_num.notna()].copy()
        tmp["__bin"] = (s_num.loc[s_num.notna()] // step_min).astype(int)
        tmp = tmp.sort_values("__bin").groupby("__bin", as_index=True).median(numeric_only=True)
        return tmp

    # Fallback: parse as datetime string.
    s_dt = pd.to_datetime(s, errors="coerce")
    if s_dt.notna().sum() == 0:
        return df.iloc[0:0]
    tmp = df.loc[s_dt.notna()].copy()
    tmp[time_col] = s_dt.loc[s_dt.notna()]
    tmp = tmp.sort_values(time_col).set_index(time_col)
    return tmp.resample(f"{step_min}min").median()


def main():
    parser = argparse.ArgumentParser(description="Build per-stay falencia summary (2/3 rule) from MIMIC parquets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--mortality_path", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--time_col", default="charttime")
    parser.add_argument("--id_col", default="stay_id")
    parser.add_argument("--mbp_col", default="mbp")
    parser.add_argument("--vaso_col", default="vasopressor_ativo")
    parser.add_argument("--lactate_col", default="lab_50813")
    parser.add_argument("--lactate_col_imputed", default="lab_50813_imputed")
    parser.add_argument("--use_imputed_lactate", action="store_true")
    parser.add_argument("--falencia_col", default="falencia")
    parser.add_argument("--step_min_45", type=int, default=5)
    parser.add_argument("--window_min_45", type=int, default=45)
    parser.add_argument("--step_min_60", type=int, default=60)
    parser.add_argument("--window_min_60", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    log_fh = open(args.log_path, "a", encoding="utf-8") if args.log_path else None

    def _log(msg: str) -> None:
        print(msg)
        if log_fh:
            log_fh.write(msg + "\n")

    files = sorted(glob.glob(str(Path(args.input_dir) / "batch_*.parquet")))
    files = [f for f in files if not f.endswith("_labels.parquet")]
    if not files:
        raise SystemExit(f"No feature parquets found in {args.input_dir} (excluding *_labels.parquet)")

    lact_col = args.lactate_col_imputed if args.use_imputed_lactate else args.lactate_col

    needed = {args.id_col, args.time_col, args.mbp_col, args.vaso_col, lact_col, args.falencia_col}
    per_stay_rows = []
    seen = set()
    dup_count = 0

    for i, path in enumerate(files, start=1):
        _log(f"[{i}/{len(files)}] {Path(path).name}")
        file_cols = set(pq.read_schema(path).names)
        read_cols = [c for c in needed if c in file_cols]
        df = pd.read_parquet(path, columns=read_cols)
        missing = needed - set(df.columns)
        if missing:
            _log(f"  [WARN] Missing columns in {Path(path).name}: {sorted(missing)}")
        for stay_id, g in df.groupby(args.id_col, sort=False):
            if stay_id in seen:
                dup_count += 1
            seen.add(stay_id)
            falencia_normal = float(np.nanmax(g[args.falencia_col].to_numpy())) if args.falencia_col in g.columns else np.nan
            fal_45 = _compute_event_any(
                g,
                args.time_col,
                args.mbp_col,
                args.vaso_col,
                lact_col,
                args.step_min_45,
                args.window_min_45,
            )
            fal_60 = _compute_event_any(
                g,
                args.time_col,
                args.mbp_col,
                args.vaso_col,
                lact_col,
                args.step_min_60,
                args.window_min_60,
            )
            per_stay_rows.append(
                {
                    args.id_col: stay_id,
                    "falencia_normal": falencia_normal,
                    "falencia_2of3_45min": int(fal_45),
                    "falencia_2of3_60min": int(fal_60),
                }
            )

    if dup_count:
        _log(f"[WARN] {dup_count} stays appeared in more than one parquet; results may be approximate.")

    out_df = pd.DataFrame(per_stay_rows).drop_duplicates(subset=[args.id_col])
    mort = pd.read_csv(args.mortality_path)
    mort_cols = [c for c in mort.columns if c.lower() in {"mortality", "death", "died"}]
    if not mort_cols:
        raise SystemExit(f"No mortality column found in {args.mortality_path}")
    mort_col = mort_cols[0]
    out_df = out_df.merge(mort[[args.id_col, mort_col]], on=args.id_col, how="left")
    out_df = out_df.rename(columns={mort_col: "mortality"})

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    _log(f"Saved: {args.out_csv} (rows={len(out_df)})")
    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()