"""One-off: evaluate a saved HF classifier checkpoint on the same test split as roberta.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    RunConfig,
    compute_metrics,
    read_dataset,
    resolve_device,
    split_dataset,
)


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(base / "outputs" / "roberta_frozen_encoder" / "saved_best_model"),
    )
    p.add_argument("--data_path", type=str, default=str(base / "Sarcasm_Headlines_Dataset_v2.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()

    cfg = RunConfig(
        data_path=args.data_path,
        model_name="roberta-base",
        output_dir="",
        seed=args.seed,
        max_length=args.max_length,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=0.0,
        num_train_epochs=0.0,
        weight_decay=0.0,
        warmup_ratio=0.0,
        device=args.device,
    )

    set_seed(cfg.seed)
    device_str = resolve_device(cfg.device)
    ckpt = Path(args.checkpoint)

    df = read_dataset(cfg.data_path)
    _, _, test_df = split_dataset(df, cfg)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
    collator = DataCollatorWithPadding(tokenizer)
    enc = tokenizer(
        test_df["headline"].tolist(),
        truncation=True,
        max_length=cfg.max_length,
    )
    test_ds = HeadlineDataset(enc, test_df["label"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    use_cuda = device_str == "cuda"
    targs = TrainingArguments(
        output_dir=str(base / ".eval_checkpoint_tmp"),
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
        fp16=use_cuda,
        no_cuda=not use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    m = trainer.evaluate(metric_key_prefix="test")

    out = {
        "checkpoint": str(ckpt.resolve()),
        "data_path": cfg.data_path,
        "seed": cfg.seed,
        "max_length": cfg.max_length,
        "device": device_str,
        "test_size": len(test_df),
        "test_pos_ratio": float(test_df["label"].mean()),
        "metrics": {k: float(v) for k, v in m.items() if k.startswith("test_")},
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
