"""
Per encoder layer L:
  v_L = mean_train(h_L | y=1) - mean_train(h_L | y=0)   [first-token hidden]
  mu_te0_L = mean_test(h_L | y=0)
  mu_te1_L = mean_test(h_L | y=1)
Report cos(v_L, mu_te0_L) and cos(v_L, mu_te1_L) (standard cosine, raw vectors).

One forward pass per batch collects all layers via output_hidden_states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from roberta import HeadlineDataset, RunConfig, read_dataset, resolve_device, split_dataset


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(base / "outputs" / "roberta_frozen_encoder" / "saved_best_model"),
    )
    p.add_argument("--data_path", type=str, default=str(base / "Sarcasm_Headlines_Dataset_v2.json"))
    p.add_argument(
        "--output_json",
        type=str,
        default=str(base / "outputs" / "roberta_frozen_encoder" / "cosine_train_diff_vs_test_means.json"),
    )
    p.add_argument("--token_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=96)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def build_split_config(args: argparse.Namespace) -> RunConfig:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    return RunConfig(
        data_path=args.data_path,
        model_name="",
        output_dir="",
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


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


@torch.no_grad()
def accumulate_layer_class_means(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    num_layers: int,
    hidden_size: int,
    token_index: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns sum_h for y=0 and y=1 per layer, and counts n0, n1 per layer (same n for all layers).
    """
    sum0 = [np.zeros(hidden_size, dtype=np.float64) for _ in range(num_layers)]
    sum1 = [np.zeros(hidden_size, dtype=np.float64) for _ in range(num_layers)]
    n0 = np.zeros(num_layers, dtype=np.int64)
    n1 = np.zeros(num_layers, dtype=np.int64)

    model.eval()
    for batch in loader:
        labels = batch.pop("labels").numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.roberta(**batch, output_hidden_states=True)
        hs_all = out.hidden_states

        m0 = labels == 0
        m1 = labels == 1
        for ell in range(num_layers):
            h = hs_all[ell + 1][:, token_index, :].float().cpu().numpy().astype(np.float64)
            if np.any(m0):
                sum0[ell] += h[m0].sum(axis=0)
                n0[ell] += int(m0.sum())
            if np.any(m1):
                sum1[ell] += h[m1].sum(axis=0)
                n1[ell] += int(m1.sum())

    return sum0, sum1, n0, n1


def main() -> None:
    args = parse_args()
    device = torch.device(resolve_device(args.device))
    cfg = build_split_config(args)
    df = read_dataset(cfg.data_path)
    train_df, _, test_df = split_dataset(df, cfg)

    base = Path(__file__).resolve().parent
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer)

    def enc_loader(frame: pd.DataFrame) -> DataLoader:
        enc = tokenizer(
            frame["headline"].tolist(),
            truncation=True,
            max_length=cfg.max_length,
        )
        ds = HeadlineDataset(enc, frame["label"].tolist())
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    train_loader = enc_loader(train_df)
    test_loader = enc_loader(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    model.to(device)
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    s0_tr, s1_tr, n0_tr, n1_tr = accumulate_layer_class_means(
        model, train_loader, device, num_layers, hidden_size, args.token_index
    )
    s0_te, s1_te, n0_te, n1_te = accumulate_layer_class_means(
        model, test_loader, device, num_layers, hidden_size, args.token_index
    )

    layers_out: List[Dict] = []
    for ell in range(num_layers):
        if n0_tr[ell] == 0 or n1_tr[ell] == 0:
            raise ValueError(f"Train missing a class at layer {ell}")
        mu_tr0 = s0_tr[ell] / n0_tr[ell]
        mu_tr1 = s1_tr[ell] / n1_tr[ell]
        v = mu_tr1 - mu_tr0

        if n0_te[ell] == 0 or n1_te[ell] == 0:
            raise ValueError(f"Test missing a class at layer {ell}")
        mu_te0 = s0_te[ell] / n0_te[ell]
        mu_te1 = s1_te[ell] / n1_te[ell]

        layers_out.append(
            {
                "layer": ell,
                "train_v_l2": float(np.linalg.norm(v)),
                "cos_v_mu_test_class0": cosine_np(v, mu_te0),
                "cos_v_mu_test_class1": cosine_np(v, mu_te1),
                "cos_mu_test_class0_mu_test_class1": cosine_np(mu_te0, mu_te1),
            }
        )

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": str(Path(cfg.data_path).resolve()),
        "split_seed": cfg.seed,
        "max_length": cfg.max_length,
        "token_index": args.token_index,
        "v_per_layer": "v = mean_train(h|y=1) - mean_train(h|y=0), same layer, first-token hidden",
        "cosine_pairs": "cos(v, mean_test(h|y=0)) and cos(v, mean_test(h|y=1)); raw vectors, not unit-normalized v only",
        "train_counts_per_layer": {"n0": int(n0_tr[0]), "n1": int(n1_tr[0])},
        "test_counts_per_layer": {"n0": int(n0_te[0]), "n1": int(n1_te[0])},
        "layers": layers_out,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report["layers"], indent=2, ensure_ascii=False))
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
