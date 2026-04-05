import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class RunConfig:
    data_path: str
    model_name: str
    output_dir: str
    seed: int
    max_length: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    num_train_epochs: float
    weight_decay: float
    warmup_ratio: float
    device: str


class HeadlineDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoBERTa binary classification (headline only)")
    parser.add_argument(
        "--data_path",
        default=str(Path(__file__).resolve().parent / "Sarcasm_Headlines_Dataset_v2.json"),
    )
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "outputs" / "roberta_split"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. 'auto' prefers CUDA when available.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    return RunConfig(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        device=args.device,
    )


def resolve_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda, but CUDA is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_dataset(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            items = json.load(f)
            for obj in items:
                records.append(
                    {
                        "headline": str(obj.get("headline", "")).strip(),
                        "label": int(obj.get("is_sarcastic", 0)),
                    }
                )
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(
                    {
                        "headline": str(obj.get("headline", "")).strip(),
                        "label": int(obj.get("is_sarcastic", 0)),
                    }
                )

    df = pd.DataFrame(records)
    df = df[df["headline"].str.len() > 0].reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame, cfg: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    texts = df["headline"]
    labels = df["label"]

    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=cfg.test_ratio,
        random_state=cfg.seed,
        stratify=labels,
    )

    val_ratio_within_train_val = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_ratio_within_train_val,
        random_state=cfg.seed,
        stratify=train_val_labels,
    )

    train_df = pd.DataFrame({"headline": train_texts.values, "label": train_labels.values})
    val_df = pd.DataFrame({"headline": val_texts.values, "label": val_labels.values})
    test_df = pd.DataFrame({"headline": test_texts.values, "label": test_labels.values})
    return train_df, val_df, test_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    accuracy = accuracy_score(labels, preds)
    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def export_predictions(path: Path, headlines: List[str], labels: np.ndarray, probs: np.ndarray):
    preds = np.argmax(probs, axis=1)
    out_df = pd.DataFrame(
        {
            "headline": headlines,
            "label_true": labels,
            "label_pred": preds,
            "prob_sarcastic": probs[:, 1],
            "error_type": np.where(labels == preds, "correct", np.where((labels == 0) & (preds == 1), "FP", "FN")),
        }
    )
    out_df.to_csv(path, index=False)


def main():
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
    use_cuda = device == "cuda"

    training_args = TrainingArguments(
        output_dir=str(Path(cfg.output_dir) / "checkpoints"),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
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
        "data_path": cfg.data_path,
        "device": device,
        "seed": cfg.seed,
        "max_length": cfg.max_length,
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

    print("Done. Training finished with train/validation/test split.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
