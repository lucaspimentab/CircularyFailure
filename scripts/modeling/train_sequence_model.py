import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tables
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HDF5StayDataset(Dataset):
    def __init__(self, h5_path: Path, split: str, max_seq_len: int, max_stays: int | None = None, seed: int = 42):
        self.h5_path = str(h5_path)
        self.split = split
        self.max_seq_len = max_seq_len
        self.static_map = None
        self._h5 = tables.open_file(self.h5_path, mode='r')

        self.windows = self._h5.root.patient_windows[split][:]
        self.stay_ids = self._h5.root.patient_windows[f"{split}_stay_ids"][:]
        self.base_dim = self._h5.root.data[split].shape[1]
        if hasattr(self._h5.root, 'static'):
            self.static_dim = self._h5.root.static[split].shape[1]
            self.static_map = {int(sid): i for i, sid in enumerate(self.stay_ids)}
        else:
            self.static_dim = 0

        if max_stays is not None and max_stays < len(self.windows):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.windows), size=max_stays, replace=False)
            self.windows = self.windows[idx]

        print(f"[INFO] Dataset {split}: {len(self.windows)} janelas (1 por stay)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, stop, stay_id = self.windows[idx]
        x = self._h5.root.data[self.split][start:stop]
        y = self._h5.root.labels[self.split][start:stop]

        if len(x) > self.max_seq_len:
            x = x[:self.max_seq_len]
            y = y[:self.max_seq_len]

        if self.static_map is not None:
            sidx = self.static_map.get(int(stay_id))
            if sidx is not None:
                static = self._h5.root.static[self.split][sidx]
            else:
                static = np.full((self.static_dim,), np.nan, dtype=np.float32)
        else:
            static = np.zeros((0,), dtype=np.float32)

        # sanitize NaNs/Infs (can appear in engineered/static or imputed data)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(static, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def collate_fn(batch):
    xs, ss, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]
    static_dim = ss[0].shape[0]
    padded_x = torch.zeros(len(xs), max_len, feat_dim)
    padded_y = torch.zeros(len(ys), max_len)
    pad_mask = torch.ones(len(xs), max_len, dtype=torch.bool)
    static_batch = torch.zeros(len(xs), static_dim)
    for i, (x, s, y) in enumerate(zip(xs, ss, ys)):
        padded_x[i, :x.shape[0]] = x
        padded_y[i, :y.shape[0]] = y.squeeze(-1)
        pad_mask[i, :x.shape[0]] = False
        if static_dim > 0:
            static_batch[i] = s
    return padded_x, static_batch, padded_y, pad_mask


def get_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "sigmoid":
        return torch.sigmoid
    raise ValueError(f"Unknown activation: {name}")


class CircEWSLikeRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dynamic: int = 500,
        hidden_static: int = 10,
        hidden_joint: int = 128,
        activ: str = "tanh",
        dropout: float = 0.0,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.static_dim = static_dim
        self.activ = get_activation(activ)
        self.dropout = nn.Dropout(dropout)

        if rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dynamic, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dynamic, batch_first=True)

        if static_dim > 0:
            self.static_proj = nn.Linear(static_dim, hidden_static)
            joint_in = hidden_dynamic + hidden_static
        else:
            self.static_proj = None
            joint_in = hidden_dynamic

        self.joint = nn.Linear(joint_in, hidden_joint)
        self.out = nn.Linear(hidden_joint, 1)

    def forward(self, x_dyn, x_static, mask):
        h, _ = self.rnn(x_dyn)

        if self.static_proj is not None:
            s = self.dropout(x_static)
            s = self.activ(self.static_proj(s))
            s = s.unsqueeze(1).expand(-1, h.shape[1], -1)
            h = torch.cat([h, s], dim=-1)

        h = self.dropout(h)
        h = self.activ(self.joint(h))
        h = self.dropout(h)
        logits = self.out(h).squeeze(-1)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[0, :, 1::2] = torch.cos(position * div_term[: (d_model // 2)])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.dtype)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        emb: int = 231,
        nhead: int = 7,
        num_layers: int = 2,
        ff_mult: int = 2,
        hidden_static: int = 10,
        activ: str = "tanh",
        dropout: float = 0.1,
        max_seq_len: int = 96,
    ):
        super().__init__()
        self.activ = get_activation(activ)
        self.dropout = nn.Dropout(dropout)
        self.static_dim = static_dim

        self.dyn_proj = nn.Linear(input_dim, emb)
        if static_dim > 0:
            self.static_proj = nn.Linear(static_dim, hidden_static)
            self.joint_proj = nn.Linear(emb + hidden_static, emb)
        else:
            self.static_proj = None
            self.joint_proj = None

        self.pos_enc = PositionalEncoding(emb, max_len=max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=nhead,
            dim_feedforward=emb * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(emb, 1)

    def forward(self, x_dyn, x_static, mask):
        h = self.dyn_proj(x_dyn)
        if self.static_proj is not None:
            s = self.dropout(x_static)
            s = self.activ(self.static_proj(s))
            s = s.unsqueeze(1).expand(-1, h.shape[1], -1)
            h = torch.cat([h, s], dim=-1)
            h = self.joint_proj(h)

        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        logits = self.out(h).squeeze(-1)
        return logits


def _safe_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else float('nan')


def _safe_auprc(y_true, y_pred):
    return average_precision_score(y_true, y_pred) if len(set(y_true)) > 1 else float('nan')


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for x, s, y, mask in tqdm(loader, desc="eval", leave=False):
            x, s, y, mask = x.to(device), s.to(device), y.to(device), mask.to(device)
            logits = model(x, s, mask)
            loss = criterion(logits, y)
            loss = loss.masked_fill(mask, 0.0)
            valid = (~mask).sum()
            total_loss += loss.sum().item()
            total_samples += valid.item()
            preds = torch.sigmoid(logits)
            all_preds.extend(preds[~mask].cpu().numpy())
            all_targets.extend(y[~mask].cpu().numpy())

    all_targets = np.asarray(all_targets)
    all_preds = np.asarray(all_preds)
    finite_mask = np.isfinite(all_targets) & np.isfinite(all_preds)
    if finite_mask.sum() < len(all_targets):
        dropped = len(all_targets) - int(finite_mask.sum())
        print(f"[WARN] Dropping {dropped} non-finite predictions/targets during eval.")
    all_targets = all_targets[finite_mask]
    all_preds = all_preds[finite_mask]

    if all_targets.size == 0:
        print("[WARN] Empty eval set after filtering non-finite values.")
        return {
            'auc': float('nan'),
            'auprc': float('nan'),
            'f1': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'loss': (total_loss / total_samples) if total_samples else float('nan'),
        }

    metrics = {
        'auc': _safe_auc(all_targets, all_preds),
        'auprc': _safe_auprc(all_targets, all_preds),
        'f1': f1_score(all_targets, np.round(all_preds), zero_division=0),
        'precision': precision_score(all_targets, np.round(all_preds), zero_division=0),
        'recall': recall_score(all_targets, np.round(all_preds), zero_division=0),
        'loss': (total_loss / total_samples) if total_samples else float('nan'),
    }
    return metrics


def _make_model_out(path: str, model_name: str) -> str:
    p = Path(path)
    stem = p.stem
    suffix = p.suffix or ".pt"
    return str(p.with_name(f"{stem}_{model_name}{suffix}"))


def train_and_evaluate(
    model,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    args,
):
    if args.early_stop_by == "auprc":
        best_score = -float('inf')
    else:
        best_score = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        for x, s, y, mask in tqdm(
            train_loader,
            desc=f"[{model_name}] epoch {epoch+1}/{args.epochs} train",
            leave=False,
        ):
            x, s, y, mask = x.to(device), s.to(device), y.to(device), mask.to(device)
            logits = model(x, s, mask)
            loss = criterion(logits, y)
            loss = loss.masked_fill(mask, 0.0)
            valid = (~mask).sum()
            loss_val = loss.sum() / valid
            if not torch.isfinite(loss_val):
                print(f"[WARN] Non-finite loss encountered in {model_name}; skipping batch.")
                optimizer.zero_grad()
                continue
            loss_val.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"[{model_name}] Val epoch {epoch+1}: "
            f"AUC={val_metrics['auc']:.4f}, F1={val_metrics['f1']:.4f}, AUPRC={val_metrics['auprc']:.4f}"
        )

        if args.early_stop_by == "auprc":
            score = val_metrics['auprc']
            improved = score > (best_score + args.min_delta)
        else:
            score = val_metrics['loss']
            improved = score < (best_score - args.min_delta)

        if improved:
            best_score = score
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[{model_name}] Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"[{model_name}] Test Final - "
        f"AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}, "
        f"AUPRC: {test_metrics['auprc']:.4f}"
    )

    return best_state, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Sequence model training on windowed HDF5.")
    parser.add_argument("--h5", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early-stop-by", type=str, default="auprc", choices=["auprc", "loss"])
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-stays", type=int, default=None)
    parser.add_argument("--max-val-stays", type=int, default=None)
    parser.add_argument("--max-test-stays", type=int, default=None)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru", "transformer"])
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list, e.g. transformer,gru,lstm")
    parser.add_argument("--hidden-dynamic", type=int, default=500)
    parser.add_argument("--hidden-static", type=int, default=10)
    parser.add_argument("--hidden-joint", type=int, default=128)
    parser.add_argument("--activ", type=str, default="tanh", choices=["tanh", "relu", "sigmoid"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--emb", type=int, default=231)
    parser.add_argument("--num-heads", type=int, default=7)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=["rmsprop", "adam"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-out", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs"))
    if args.h5 is None:
        args.h5 = str(output_dir / "preprocess" / "h5" / "dataset_windowed.h5")


    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise SystemExit(f"HDF5 not found: {h5_path}")

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if args.model_out is None:
        args.model_out = str(output_dir / f"models/{{MODEL}}_seed{args.seed}.pt")
    if args.metrics_csv is None:
        args.metrics_csv = str(output_dir / "models" / "metrics_all.csv")

    train_ds = val_ds = test_ds = None
    try:
        train_ds = HDF5StayDataset(h5_path, 'train', args.max_seq_len, args.max_train_stays, args.seed)
        val_ds = HDF5StayDataset(h5_path, 'val', args.max_seq_len, args.max_val_stays, args.seed)
        test_ds = HDF5StayDataset(h5_path, 'test', args.max_seq_len, args.max_test_stays, args.seed)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

        with tables.open_file(h5_path, mode='r') as f:
            base_dim = f.root.data['train'].shape[1]
            static_dim = f.root.static['train'].shape[1] if hasattr(f.root, 'static') else 0

        if args.models:
            models_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        else:
            models_list = [args.model]
        valid_models = {"lstm", "gru", "transformer"}
        for m in models_list:
            if m not in valid_models:
                raise SystemExit(f"Unknown model '{m}'. Valid: {sorted(valid_models)}")

        criterion = nn.BCEWithLogitsLoss(reduction='none')

        for model_name in models_list:
            set_seed(args.seed)
            model = None
            optimizer = None
            best_state = None
            test_metrics = {
                "auc": float("nan"),
                "auprc": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "loss": float("nan"),
            }
            try:
                if model_name == "transformer":
                    if args.emb % args.num_heads != 0:
                        raise SystemExit(f"--emb ({args.emb}) must be divisible by --num-heads ({args.num_heads})")
                    model = TransformerModel(
                        input_dim=base_dim,
                        static_dim=static_dim,
                        emb=args.emb,
                        nhead=args.num_heads,
                        num_layers=args.num_layers,
                        ff_mult=args.ff_mult,
                        hidden_static=args.hidden_static,
                        activ=args.activ,
                        dropout=args.dropout,
                        max_seq_len=args.max_seq_len,
                    ).to(device)
                else:
                    model = CircEWSLikeRNN(
                        input_dim=base_dim,
                        static_dim=static_dim,
                        hidden_dynamic=args.hidden_dynamic,
                        hidden_static=args.hidden_static,
                        hidden_joint=args.hidden_joint,
                        activ=args.activ,
                        dropout=args.dropout,
                        rnn_type=model_name,
                    ).to(device)

                if args.optimizer == "rmsprop":
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

                best_state, test_metrics = train_and_evaluate(
                    model,
                    model_name.upper(),
                    train_loader,
                    val_loader,
                    test_loader,
                    criterion,
                    optimizer,
                    device,
                    args,
                )
            except Exception as exc:
                print(f"[ERROR] Model {model_name} failed: {exc}")
            finally:
                if best_state is not None:
                    out_path = args.model_out.replace("{MODEL}", model_name.upper())
                    out_path = _make_model_out(out_path, model_name.lower())
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, out_path)

                # Save metrics after each model
                row = {**test_metrics, "seed": args.seed, "model": model_name}
                Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
                import csv
                file_exists = Path(args.metrics_csv).exists()
                with Path(args.metrics_csv).open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
                print(f"Metrics saved to: {args.metrics_csv}")

                # Cleanup to avoid OOM across models
                del model, optimizer, best_state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if train_ds is not None:
            train_ds.close()
        if val_ds is not None:
            val_ds.close()
        if test_ds is not None:
            test_ds.close()


if __name__ == "__main__":
    main()

