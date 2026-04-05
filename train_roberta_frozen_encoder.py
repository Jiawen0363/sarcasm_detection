"""
Train only the classification head on top of a frozen pretrained RoBERTa encoder
(same data split as roberta.py). Encoder weights stay identical to model_name (e.g. roberta-base).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from roberta import (
    HeadlineDataset,
    build_config,
    compute_metrics,
    export_predictions,
    read_dataset,
    resolve_device,
    split_dataset,
)


def freeze_roberta_encoder(model: AutoModelForSequenceClassification) -> None:
    for name, p in model.named_parameters():
        if name.startswith("roberta."):
            p.requires_grad = False


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def prune_checkpoints_keep_only_step(checkpoints_root: Path, step: int) -> None:
    keep = checkpoints_root / f"checkpoint-{step}"
    if not keep.is_dir():
        print(f"[retain_only] {keep} not found — skip pruning (other checkpoints left as-is).")
        return
    removed = 0
    for p in sorted(checkpoints_root.glob("checkpoint-*")):
        if not p.is_dir():
            continue
        if p.name == f"checkpoint-{step}":
            continue
        shutil.rmtree(p)
        removed += 1
    print(f"[retain_only] Kept {keep.name}; removed {removed} other checkpoint dir(s).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen RoBERTa encoder + train classifier head only")
    base = Path(__file__).resolve().parent
    p.add_argument(
        "--data_path",
        type=str,
        default=str(base / "Sarcasm_Headlines_Dataset_v2.json"),
    )
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(base / "outputs" / "roberta_frozen_encoder"),
        help="Where to write checkpoints, metrics, test_predictions.csv",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=96)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Often higher than full fine-tuning when only the head trains.",
    )
    p.add_argument(
        "--num_train_epochs",
        type=float,
        default=10.0,
        help="Head-only may need more epochs; best checkpoint by val F1.",
    )
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. 'auto' prefers CUDA when available.",
    )
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Max checkpoints to retain during training (Trainer-side). Default None keeps all epoch saves.",
    )
    p.add_argument(
        "--retain_only_checkpoint_step",
        type=int,
        default=716,
        help="After training, delete every checkpoints/checkpoint-* except checkpoint-{step}. "
        "Set to -1 to disable. Requires that step to exist (e.g. end of epoch 1 with batch 32 on ~23k samples).",
    )
    p.add_argument(
        "--snapshot_best_dir",
        type=str,
        default="saved_best_model",
        help="Relative to output_dir: save in-memory model (val-best after train) here before checkpoint pruning, "
        "so you still have best weights if best step != retain_only step.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    device = resolve_device(cfg.device)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    df = read_dataset(cfg.data_path)
    train_df, val_df, test_df = split_dataset(df, cfg)

    split_stat = {
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "train_pos_ratio": float(train_df["label"].mean()),
        "val_pos_ratio": float(val_df["label"].mean()),
        "test_pos_ratio": float(test_df["label"].mean()),
    }
    with open(Path(cfg.output_dir) / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(split_stat, f, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_enc = tokenizer(train_df["headline"].tolist(), truncation=True, max_length=cfg.max_length)
    val_enc = tokenizer(val_df["headline"].tolist(), truncation=True, max_length=cfg.max_length)
    test_enc = tokenizer(test_df["headline"].tolist(), truncation=True, max_length=cfg.max_length)

    train_ds = HeadlineDataset(train_enc, train_df["label"].tolist())
    val_ds = HeadlineDataset(val_enc, val_df["label"].tolist())
    test_ds = HeadlineDataset(test_enc, test_df["label"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    freeze_roberta_encoder(model)
    trainable, total = count_trainable_params(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    use_cuda = device == "cuda"

    training_args = TrainingArguments(
        output_dir=str(Path(cfg.output_dir) / "checkpoints"),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        logging_strategy="epoch",
        report_to="none",
        fp16=use_cuda,
        dataloader_num_workers=0,
        seed=cfg.seed,
        no_cuda=not use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    out_root = Path(cfg.output_dir)
    ckpt_root = out_root / "checkpoints"
    if args.retain_only_checkpoint_step >= 0:
        snap = out_root / args.snapshot_best_dir
        snap.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(snap)
        tokenizer.save_pretrained(snap)
        print(f"Saved in-memory (val-best) model to {snap.resolve()}")
        prune_checkpoints_keep_only_step(ckpt_root, args.retain_only_checkpoint_step)

    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    test_pred_output = trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_pred_output.predictions), dim=-1).numpy()
    export_predictions(
        path=Path(cfg.output_dir) / "test_predictions.csv",
        headlines=test_df["headline"].tolist(),
        labels=test_df["label"].to_numpy(),
        probs=test_probs,
    )

    result = {
        "model_name": cfg.model_name,
        "frozen_encoder": True,
        "trainable_param_count": trainable,
        "total_param_count": total,
        "data_path": cfg.data_path,
        "device": device,
        "seed": cfg.seed,
        "max_length": cfg.max_length,
        "head_learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "retain_only_checkpoint_step": args.retain_only_checkpoint_step,
        "snapshot_best_dir": str(out_root / args.snapshot_best_dir)
        if args.retain_only_checkpoint_step >= 0
        else None,
        "split": split_stat,
        "val": {
            "accuracy": float(val_metrics["val_accuracy"]),
            "precision": float(val_metrics["val_precision"]),
            "recall": float(val_metrics["val_recall"]),
            "f1": float(val_metrics["val_f1"]),
            "roc_auc": float(val_metrics["val_roc_auc"]),
        },
        "test": {
            "accuracy": float(test_metrics["test_accuracy"]),
            "precision": float(test_metrics["test_precision"]),
            "recall": float(test_metrics["test_recall"]),
            "f1": float(test_metrics["test_f1"]),
            "roc_auc": float(test_metrics["test_roc_auc"]),
        },
    }
    with open(Path(cfg.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Done. Frozen encoder; classifier head trained.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
