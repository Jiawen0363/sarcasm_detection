"""
Steering experiment (fine-tuned RoBERTa, LiReFs-style mean-diff direction):

1. On TRAIN split: extract layer-L first-token hidden from model.roberta (same checkpoint).
2. v = mean(h | label=1) - mean(h | label=0), then L2-normalize.
3. On TEST: register forward hook on roberta.encoder.layer[L] to add alpha * v at token 0 only.
4. Report pred_sarcasm rate (argmax==1) overall and stratified by true label; optional mean prob.

Matches roberta.py data split / tokenizer / max_length when given the same flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from roberta import HeadlineDataset, RunConfig, read_dataset, resolve_device, split_dataset


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Layer steering via train mean-diff, test pred rate")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(base / "outputs" / "roberta_frozen_encoder" / "saved_best_model"),
    )
    p.add_argument("--data_path", type=str, default=str(base / "Sarcasm_Headlines_Dataset_v2.json"))
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(base / "outputs" / "roberta_frozen_encoder" / "steer_layer9"),
    )
    p.add_argument("--layer", type=int, default=9, help="Encoder layer index 0..num_layers-1 (default 9).")
    p.add_argument("--token_index", type=int, default=0, help="Where to read/add steering (0 = <s> / first).")
    p.add_argument(
        "--alphas",
        type=str,
        default="-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8",
        help="Comma-separated steering strengths (alpha * v_hat added to hidden; negative reverses direction).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=96)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--save_v", action="store_true", help="Save v_hat as steer_v_layer{L}.npy in output_dir.")
    return p.parse_args()


def build_split_config(args: argparse.Namespace) -> RunConfig:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    return RunConfig(
        data_path=args.data_path,
        model_name="",
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


def parse_alphas(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("--alphas must contain at least one value")
    return sorted(set(out))


@torch.no_grad()
def collect_layer_first_token(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    token_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns X (N, H) float32, y (N,) int64 from model.roberta hidden_states[layer_idx+1]."""
    model.eval()
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for batch in loader:
        labels = batch.pop("labels").numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.roberta(**batch, output_hidden_states=True)
        hs = out.hidden_states[layer_idx + 1]
        vec = hs[:, token_index, :].detach().float().cpu().numpy()
        xs.append(vec)
        ys.append(labels)

    X = np.vstack(xs)
    y = np.concatenate(ys, axis=0)
    return X, y


def projection_stats(proj: np.ndarray, y: np.ndarray) -> Dict:
    """proj[i] = h_i^T v_hat (v_hat unit). y in {0,1}."""
    out: Dict = {}
    for c, name in [(0, "true_not_sarcastic_y0"), (1, "true_sarcastic_y1")]:
        m = y == c
        if not np.any(m):
            out[name] = None
            continue
        p = proj[m]
        out[name] = {
            "n": int(m.sum()),
            "mean": float(np.mean(p)),
            "std": float(np.std(p)),
            "min": float(np.min(p)),
            "max": float(np.max(p)),
        }
    m0 = out.get("true_not_sarcastic_y0")
    m1 = out.get("true_sarcastic_y1")
    if m0 is not None and m1 is not None:
        diff = m1["mean"] - m0["mean"]
        out["mean_proj_difference_y1_minus_y0"] = float(diff)
        out[
            "interpretation"
        ] = "v_hat ∝ train mean(h|y=1)-mean(h|y=0). On test, expect mean(proj|y=1) > mean(proj|y=0) if geometry aligns."
        out["sarcastic_higher_projection"] = bool(diff > 0)
    return out


def compute_mean_diff_unit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """v = mu_pos - mu_neg for label 1 vs 0; return unit v and diagnostics."""
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("Train split must contain both classes for mean-diff steering.")
    X64 = X.astype(np.float64)
    mu0 = X64[y == 0].mean(axis=0)
    mu1 = X64[y == 1].mean(axis=0)
    v = mu1 - mu0
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("Mean-diff vector is (near) zero; cannot normalize.")
    v_hat = (v / norm).astype(np.float32)
    stats = {
        "train_n": int(len(y)),
        "train_n_class_0": int((y == 0).sum()),
        "train_n_class_1": int((y == 1).sum()),
        "mean_diff_l2": norm,
        "cosine_mu0_mu1": float(
            np.dot(mu0, mu1) / (np.linalg.norm(mu0) * np.linalg.norm(mu1) + 1e-12)
        ),
    }
    return v_hat, stats


def make_layer_output_hook(
    alpha: float,
    v_hat: np.ndarray,
    token_index: int,
):
    v_tensor = torch.from_numpy(v_hat)

    def hook(module: nn.Module, inputs: Tuple, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = ()
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"Unexpected layer output type: {type(h)}")
        delta = (alpha * v_tensor).to(device=h.device, dtype=h.dtype)
        h_new = h.clone()
        h_new[:, token_index, :] = h_new[:, token_index, :] + delta
        if isinstance(output, tuple):
            return (h_new,) + rest
        return h_new

    return hook  # for register_forward_hook


@torch.no_grad()
def evaluate_steered(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    layer_module: nn.Module,
    alpha: float,
    v_hat: np.ndarray,
    token_index: int,
) -> Dict[str, float]:
    model.eval()
    hook_fn = make_layer_output_hook(alpha, v_hat, token_index)
    handle = layer_module.register_forward_hook(hook_fn)

    n = 0
    n_pred_sarc = 0
    n_true0 = 0
    n_true1 = 0
    pred_sarc_given_true0 = 0
    pred_sarc_given_true1 = 0
    pred_not_sarc_given_true0 = 0
    pred_not_sarc_given_true1 = 0
    sum_prob_sarc = 0.0
    n_correct = 0

    try:
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            p_sarc = probs[:, 1]
            pred = logits.argmax(dim=-1)

            n += labels.numel()
            n_correct += int((pred == labels).sum().item())
            n_pred_sarc += int((pred == 1).sum().item())
            sum_prob_sarc += float(p_sarc.sum().item())

            m0 = labels == 0
            m1 = labels == 1
            n_true0 += int(m0.sum().item())
            n_true1 += int(m1.sum().item())
            pred_sarc_given_true0 += int((pred[m0] == 1).sum().item())
            pred_sarc_given_true1 += int((pred[m1] == 1).sum().item())
            pred_not_sarc_given_true0 += int((pred[m0] == 0).sum().item())
            pred_not_sarc_given_true1 += int((pred[m1] == 0).sum().item())
    finally:
        handle.remove()

    return {
        "n": float(n),
        "pred_sarcasm_rate": n_pred_sarc / n if n else 0.0,
        "mean_prob_sarcastic": sum_prob_sarc / n if n else 0.0,
        "pred_sarcasm_rate_given_true_not_sarcastic": pred_sarc_given_true0 / n_true0 if n_true0 else 0.0,
        "pred_sarcasm_rate_given_true_sarcastic": pred_sarc_given_true1 / n_true1 if n_true1 else 0.0,
        "pred_not_sarcasm_rate_given_true_not_sarcastic": pred_not_sarc_given_true0 / n_true0 if n_true0 else 0.0,
        "pred_not_sarcasm_rate_given_true_sarcastic": pred_not_sarc_given_true1 / n_true1 if n_true1 else 0.0,
        "accuracy": n_correct / n if n else 0.0,
    }


def main() -> None:
    args = parse_args()
    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    cfg = build_split_config(args)
    df = read_dataset(cfg.data_path)
    train_df, _, test_df = split_dataset(df, cfg)

    num_layers = None  # set after load

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer)

    def make_loader(frame: pd.DataFrame) -> DataLoader:
        enc = tokenizer(
            frame["headline"].tolist(),
            truncation=True,
            max_length=cfg.max_length,
        )
        ds = HeadlineDataset(enc, frame["label"].tolist())
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    train_loader = make_loader(train_df)
    test_loader = make_loader(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    model.to(device)

    num_layers = model.config.num_hidden_layers
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"--layer must be in [0, {num_layers - 1}], got {args.layer}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr = collect_layer_first_token(
        model, train_loader, device, args.layer, args.token_index
    )
    v_hat, v_stats = compute_mean_diff_unit(X_tr, y_tr)

    X_te, y_te = collect_layer_first_token(
        model, test_loader, device, args.layer, args.token_index
    )
    proj_te = X_te.astype(np.float32) @ v_hat
    test_projection = {
        "definition": "proj = h^T v_hat on TEST set, same layer/token as v_hat; v_hat unit L2",
        "all_test_mean_proj": float(np.mean(proj_te)),
        **projection_stats(proj_te, y_te),
    }

    if args.save_v:
        np.save(out_dir / f"steer_v_layer{args.layer}.npy", v_hat)

    layer_module = model.roberta.encoder.layer[args.layer]
    alphas = parse_alphas(args.alphas)
    rows = []
    for a in alphas:
        m = evaluate_steered(model, test_loader, device, layer_module, a, v_hat, args.token_index)
        rows.append({"alpha": a, **m})

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": str(Path(cfg.data_path).resolve()),
        "split_seed": cfg.seed,
        "max_length": cfg.max_length,
        "layer": args.layer,
        "token_index": args.token_index,
        "steering": "add alpha * v_hat to encoder.layer[L] output at token_index (forward hook)",
        "v_definition": "train mean(h|y=1) - mean(h|y=0), L2 normalize; h = roberta hidden_states[L+1][:,token_index,:]",
        "v_stats": v_stats,
        "test_projection_on_train_v_hat": test_projection,
        "test_metrics_by_alpha": rows,
    }
    out_json = out_dir / "steering_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
