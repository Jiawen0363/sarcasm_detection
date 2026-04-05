#!/usr/bin/env python3
"""Quantitative error analysis: FP/FN by model + overlap error-type agreement."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LR_JSON = ROOT / "lr_fp_fn_errors.json"
ROBERTA_CSV = ROOT / "outputs" / "roberta_split" / "test_predictions.csv"
OUT_DIR = ROOT / "outputs" / "roberta_split" / "error_analysis"


def load_baseline_maps(path: Path) -> tuple[dict[str, str], int, int]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    headline_to_et: dict[str, str] = {}
    for h in data.get("false_positives", []):
        headline_to_et[h["headline"].strip()] = "FP"
    for h in data.get("false_negatives", []):
        headline_to_et[h["headline"].strip()] = "FN"
    fp = sum(1 for v in headline_to_et.values() if v == "FP")
    fn = sum(1 for v in headline_to_et.values() if v == "FN")
    return headline_to_et, fp, fn


def load_roberta_maps(path: Path) -> tuple[dict[str, str], int, int]:
    headline_to_et: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            et = row.get("error_type", "").strip()
            if et not in ("FP", "FN"):
                continue
            headline_to_et[row["headline"].strip()] = et
    fp = sum(1 for v in headline_to_et.values() if v == "FP")
    fn = sum(1 for v in headline_to_et.values() if v == "FN")
    return headline_to_et, fp, fn


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lr_map, lr_fp, lr_fn = load_baseline_maps(LR_JSON)
    rb_map, rb_fp, rb_fn = load_roberta_maps(ROBERTA_CSV)

    overlap = set(lr_map) & set(rb_map)
    both_fp = sum(1 for h in overlap if lr_map[h] == "FP" and rb_map[h] == "FP")
    both_fn = sum(1 for h in overlap if lr_map[h] == "FN" and rb_map[h] == "FN")
    lr_fp_rb_fn = sum(1 for h in overlap if lr_map[h] == "FP" and rb_map[h] == "FN")
    lr_fn_rb_fp = sum(1 for h in overlap if lr_map[h] == "FN" and rb_map[h] == "FP")
    same_dir = both_fp + both_fn

    stats = {
        "lr_total_errors": lr_fp + lr_fn,
        "lr_fp": lr_fp,
        "lr_fn": lr_fn,
        "roberta_total_errors": rb_fp + rb_fn,
        "roberta_fp": rb_fp,
        "roberta_fn": rb_fn,
        "overlap_count": len(overlap),
        "overlap_both_fp": both_fp,
        "overlap_both_fn": both_fn,
        "overlap_lr_fp_roberta_fn": lr_fp_rb_fn,
        "overlap_lr_fn_roberta_fp": lr_fn_rb_fp,
        "overlap_same_direction": same_dir,
        "overlap_same_direction_frac_of_overlap": round(same_dir / len(overlap), 4)
        if overlap
        else 0.0,
    }
    with (OUT_DIR / "confusion_direction_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # --- Figure 1: grouped bars FP vs FN per model ---
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    width = 0.35
    lr_vals = [lr_fp, lr_fn]
    rb_vals = [rb_fp, rb_fn]
    bars1 = ax.bar(x - width / 2, lr_vals, width, label="Baseline (LR)", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, rb_vals, width, label="RoBERTa (split)", color="#DD8452")
    ax.set_ylabel("Count")
    ax.set_title("Test errors by confusion type")
    ax.set_xticks(x)
    ax.set_xticklabels(["False positive (pred sarcastic)", "False negative (pred not sarcastic)"])
    ax.legend()
    ax.bar_label(bars1, padding=2)
    ax.bar_label(bars2, padding=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_fp_fn_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: overlap 2x2 (error-type agreement on stubborn errors) ---
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    mat = np.array([[both_fp, lr_fn_rb_fp], [lr_fp_rb_fn, both_fn]], dtype=float)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["RoBERTa FP", "RoBERTa FN"])
    ax.set_yticklabels(["LR FP", "LR FN"])
    ax.set_xlabel("RoBERTa error type")
    ax.set_ylabel("LR error type")
    ax.set_title(f"Stubborn errors (n={len(overlap)}): LR ∩ RoBERTa")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stubborn_errors_lr_vs_roberta_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: stacked bar — share of overlap by agreement type ---
    labels = ["Both FP", "Both FN", "LR FP → RoBERTa FN", "LR FN → RoBERTa FP"]
    counts = [both_fp, both_fn, lr_fp_rb_fn, lr_fn_rb_fp]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Stubborn errors: how LR and RoBERTa error types align")
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.02, str(c), ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stubborn_errors_direction_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote figures and stats under: {OUT_DIR}")


if __name__ == "__main__":
    main()
