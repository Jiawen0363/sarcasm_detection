"""
Extract first-token hidden states per layer on the test split (same as roberta.py),
then PCA(2) per layer and save scatter plots (sarcastic vs non-sarcastic).

Does not store full [N, T, 768] tensors; only accumulates [N, 768] per layer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    RobertaModel,
)

from roberta import HeadlineDataset, RunConfig, read_dataset, resolve_device, split_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA visualization of per-layer first-token RoBERTa hidden states")
    base = Path(__file__).resolve().parent
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(base / "outputs" / "roberta_split" / "checkpoints" / "checkpoint-179"),
    )
    p.add_argument(
        "--data_path",
        type=str,
        default=str(base / "Sarcasm_Headlines_Dataset_v2.json"),
    )
    p.add_argument("--output_dir", type=str, default=str(base / "outputs" / "roberta_split" / "pca_first_token"))
    p.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="Pretrained weights id (HuggingFace). Used for tokenizer when --pretrained; else only fallback.",
    )
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained RobertaModel (model_name), not fine-tuned checkpoint. Same test split and PCA pipeline.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=96)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save layer_00.npz ... with X (N,768) and y for reproducibility",
    )
    p.add_argument(
        "--predictions_csv",
        type=str,
        default="",
        help=(
            "If set, CSV with columns headline + error_type (correct|FP|FN) for the same test split. "
            "FP/FN use distinct colors; correct points are faint by gold label. "
            "Does not load unless this flag is non-empty (avoids overwriting baseline PCA figures)."
        ),
    )
    return p.parse_args()


def build_split_config(args: argparse.Namespace) -> RunConfig:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    return RunConfig(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=str(Path(args.output_dir).parent),
        seed=args.seed,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=0.0,
        num_train_epochs=0.0,
        weight_decay=0.0,
        warmup_ratio=0.0,
        device=args.device,
    )


def _roberta_forward_hidden_states(
    encoder: Union[RobertaModel, torch.nn.Module], batch: Dict[str, torch.Tensor]
):
    if hasattr(encoder, "roberta"):
        return encoder.roberta(**batch, output_hidden_states=True)
    return encoder(**batch, output_hidden_states=True)


@torch.no_grad()
def collect_first_token_hiddens(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_layers: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Returns:
        layers_vecs: list length num_layers, each array (N, hidden_size)
        labels: (N,) int64
    """
    layer_chunks: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    label_chunks: List[np.ndarray] = []

    encoder.eval()
    for batch in dataloader:
        labels = batch.pop("labels").numpy()
        label_chunks.append(labels)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = _roberta_forward_hidden_states(encoder, batch)
        # hidden_states[0] = embeddings; hidden_states[1+i] = after layer i
        hs = out.hidden_states
        for layer_idx in range(num_layers):
            first = hs[layer_idx + 1][:, 0, :].detach().float().cpu().numpy()
            layer_chunks[layer_idx].append(first)

    y = np.concatenate(label_chunks, axis=0)
    layers_vecs = [np.vstack(ch) for ch in layer_chunks]
    return layers_vecs, y


def load_error_types_from_predictions_csv(csv_path: Path, headlines: List[str]) -> np.ndarray:
    """Return length-N array of 'correct' | 'FP' | 'FN' in ``headlines`` order."""
    by_h: Dict[str, str] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row["headline"].strip()
            et = row["error_type"].strip()
            if et not in ("correct", "FP", "FN"):
                continue
            by_h[h] = et
    out: List[str] = []
    missing: List[str] = []
    for h in headlines:
        key = h.strip() if isinstance(h, str) else str(h).strip()
        if key not in by_h:
            missing.append(key)
        else:
            out.append(by_h[key])
    if missing:
        raise KeyError(f"{len(missing)} test headlines missing from predictions CSV (first: {missing[0]!r})")
    return np.array(out, dtype=object)


def plot_pca_layer(
    X: np.ndarray,
    y: np.ndarray,
    layer_idx: int,
    out_path: Path,
    evr: Tuple[float, float],
    title_suffix: str = "",
    error_types: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    if error_types is None:
        sarc = y == 1
        non = y == 0
        ax.scatter(X[non, 0], X[non, 1], c="#4C72B0", s=8, alpha=0.45, label="Not sarcastic (0)")
        ax.scatter(X[sarc, 0], X[sarc, 1], c="#DD8452", s=8, alpha=0.45, label="Sarcastic (1)")
    else:
        assert len(error_types) == len(y)
        ok = error_types == "correct"
        fp = error_types == "FP"
        fn = error_types == "FN"
        # Correct: gold-label colors, de-emphasized
        c0 = ok & (y == 0)
        c1 = ok & (y == 1)
        ax.scatter(
            X[c0, 0],
            X[c0, 1],
            c="#4C72B0",
            s=7,
            alpha=0.22,
            label="Correct (gold 0)",
            rasterized=True,
        )
        ax.scatter(
            X[c1, 0],
            X[c1, 1],
            c="#DD8452",
            s=7,
            alpha=0.22,
            label="Correct (gold 1)",
            rasterized=True,
        )
        ax.scatter(
            X[fp, 0],
            X[fp, 1],
            c="#C44E52",
            s=22,
            alpha=0.9,
            edgecolors="#2F2F2F",
            linewidths=0.25,
            label="FP (pred sarcastic)",
            zorder=5,
        )
        ax.scatter(
            X[fn, 0],
            X[fn, 1],
            c="#2CA02C",
            s=22,
            alpha=0.9,
            edgecolors="#2F2F2F",
            linewidths=0.25,
            label="FN (pred not sarcastic)",
            zorder=5,
        )
        title_suffix = f"{title_suffix} — RoBERTa errors"
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Layer {layer_idx} — first-token hidden → PCA(2){title_suffix}")
    ax.legend(loc="best", fontsize=7, markerscale=1.2)
    ax.text(
        0.02,
        0.98,
        f"EVR: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    base = Path(__file__).resolve().parent
    default_out = (base / "outputs" / "roberta_split" / "pca_first_token").resolve()
    if args.pretrained and Path(args.output_dir).resolve() == default_out:
        args.output_dir = str(default_out.parent / "pca_first_token_pretrained")

    cfg = build_split_config(args)
    df = read_dataset(cfg.data_path)
    _, _, test_df = split_dataset(df, cfg)

    pred_csv_arg = (args.predictions_csv or "").strip()
    error_types: Optional[np.ndarray] = None
    pred_csv_path: Optional[Path] = None
    if pred_csv_arg:
        pred_csv_path = Path(pred_csv_arg)
        if not pred_csv_path.is_file():
            raise FileNotFoundError(f"--predictions_csv not found: {pred_csv_path.resolve()}")
        error_types = load_error_types_from_predictions_csv(pred_csv_path, test_df["headline"].tolist())

    tok_source = args.model_name if args.pretrained else args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer)
    enc = tokenizer(
        test_df["headline"].tolist(),
        truncation=True,
        max_length=cfg.max_length,
    )
    ds = HeadlineDataset(enc, test_df["label"].tolist())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    if args.pretrained:
        encoder = RobertaModel.from_pretrained(args.model_name)
        weights_desc = f"pretrained:{args.model_name}"
    else:
        encoder = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
        weights_desc = str(Path(args.checkpoint).resolve())

    encoder.to(device)

    num_layers = encoder.config.num_hidden_layers
    hidden_size = encoder.config.hidden_size

    title_suffix = " (pretrained RoBERTa)" if args.pretrained else ""
    if error_types is not None and pred_csv_path is not None:
        title_suffix = f"{title_suffix} (FP/FN from {pred_csv_path.name})"

    layers_vecs, y = collect_first_token_hiddens(encoder, loader, device, num_layers)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    for layer_idx, X in enumerate(layers_vecs):
        assert X.shape == (len(y), hidden_size)
        pca = PCA(n_components=2, random_state=cfg.seed)
        Z = pca.fit_transform(X)
        evr = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))
        png_path = out_root / f"layer_{layer_idx:02d}.png"
        plot_pca_layer(Z, y, layer_idx, png_path, evr, title_suffix=title_suffix, error_types=error_types)
        row = {
            "layer": layer_idx,
            "n": int(len(y)),
            "n_class_0": int((y == 0).sum()),
            "n_class_1": int((y == 1).sum()),
            "evr_pc1": evr[0],
            "evr_pc2": evr[1],
            "figure": str(png_path.resolve()),
        }
        manifest.append(row)
        if args.save_npz:
            np.savez_compressed(out_root / f"layer_{layer_idx:02d}.npz", X=X.astype(np.float32), y=y)

    meta = {
        "weights": weights_desc,
        "pretrained_only": bool(args.pretrained),
        "checkpoint": str(Path(args.checkpoint).resolve()) if not args.pretrained else None,
        "model_name": args.model_name,
        "data_path": str(Path(cfg.data_path).resolve()),
        "split_seed": cfg.seed,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
        "max_length": cfg.max_length,
        "pooling": "first_token_index_0",
        "pca": "sklearn PCA n_components=2, fit on all test samples (both classes)",
        "predictions_csv": str(pred_csv_path.resolve()) if pred_csv_path is not None else None,
        "n_fp_highlighted": int((error_types == "FP").sum()) if error_types is not None else None,
        "n_fn_highlighted": int((error_types == "FN").sum()) if error_types is not None else None,
        "layers": manifest,
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(manifest)} figures to {out_root}")
    print(json.dumps(meta["layers"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
