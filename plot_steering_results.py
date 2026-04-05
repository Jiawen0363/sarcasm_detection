"""
Visualize steering_results.json.

Default dashboard (--layout lines):
  1) Overall: pred_sarcasm_rate, mean_prob_sarcastic, accuracy vs α (baseline + best-acc markers)
  2) Stratified: P(pred sarc | true ¬sarc) vs P(pred sarc | true sarc)
  3) Δ vs baseline α: same five metrics as lines (replaces diverging heatmap)
  4) Optional stacked bars when n_alpha is small (see --bars)

Heatmap layout (--layout heatmap): original A–D panels + trajectory figure E.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec


def checkpoint_display_name(checkpoint: str) -> str:
    """Use parent folder for typical HF save dir (e.g. .../roberta_frozen_encoder/saved_best_model)."""
    p = Path(checkpoint)
    if p.name == "saved_best_model" and p.parent.name:
        return p.parent.name
    return p.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Steering metrics dashboard from steering_results.json")
    base = Path(__file__).resolve().parent
    p.add_argument(
        "--input_json",
        type=str,
        default=str(base / "outputs" / "roberta_frozen_encoder" / "steer_layer9" / "steering_results.json"),
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output .png. Default: <json_dir>/steering_dashboard.png",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--layout",
        choices=("lines", "heatmap"),
        default="lines",
        help="lines: α on x-axis, curves (default). heatmap: matrix + stacked bars.",
    )
    p.add_argument(
        "--bars",
        action="store_true",
        help="With --layout lines: add a bottom row of 100% stacked bars (C/D style).",
    )
    p.add_argument(
        "--lines",
        action="store_true",
        help="Also write steering_curves.png (three-panel line summary).",
    )
    return p.parse_args()


def load_arrays(data: dict):
    rows = sorted(data["test_metrics_by_alpha"], key=lambda r: float(r["alpha"]))
    a = np.array([float(r["alpha"]) for r in rows])
    M = np.vstack(
        [
            np.array([float(r["pred_sarcasm_rate"]) for r in rows]),
            np.array([float(r["mean_prob_sarcastic"]) for r in rows]),
            np.array([float(r["pred_sarcasm_rate_given_true_not_sarcastic"]) for r in rows]),
            np.array([float(r["pred_sarcasm_rate_given_true_sarcastic"]) for r in rows]),
            np.array([float(r.get("accuracy", float("nan"))) for r in rows]),
        ]
    )
    p_sarc_y0 = np.array([float(r["pred_sarcasm_rate_given_true_not_sarcastic"]) for r in rows])
    p_sarc_y1 = np.array([float(r["pred_sarcasm_rate_given_true_sarcastic"]) for r in rows])
    p_not_y0 = np.array([float(r.get("pred_not_sarcasm_rate_given_true_not_sarcastic", 1.0 - s)) for r, s in zip(rows, p_sarc_y0)])
    p_not_y1 = np.array([float(r.get("pred_not_sarcasm_rate_given_true_sarcastic", 1.0 - s)) for r, s in zip(rows, p_sarc_y1)])
    pred_rate = M[0]
    mean_p = M[1]
    acc = M[4]
    row_labels = [
        "pred_sarcasm_rate",
        "mean_prob_sarcastic",
        "P(pred sarc | true ¬sarc)",
        "P(pred sarc | true sarc)",
        "accuracy",
    ]
    return a, M, row_labels, p_not_y0, p_sarc_y0, p_not_y1, p_sarc_y1, pred_rate, mean_p, acc


def _plot_trajectory_figure(out: Path, dpi: int, a, pred_rate, acc, n_a) -> None:
    out_traj = out.with_name(out.stem + "_trajectory.png")
    fig2, ax = plt.subplots(figsize=(6.5, 5.5), dpi=dpi)
    sc = ax.scatter(
        pred_rate,
        acc,
        c=a,
        cmap="coolwarm",
        s=55,
        zorder=3,
        edgecolors="k",
        linewidths=0.35,
    )
    ax.plot(pred_rate, acc, color="gray", linewidth=0.8, alpha=0.5, zorder=1)
    for i, al in enumerate(a):
        if i % max(1, n_a // 6) == 0 or al == 0 or i == n_a - 1:
            ax.annotate(f"{al:g}", (pred_rate[i], acc[i]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("pred_sarcasm_rate (overall)")
    ax.set_ylabel("accuracy")
    ax.set_title("Trajectory: overall pred sarcasm rate vs accuracy (color = α)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.colorbar(sc, ax=ax, label=r"$\alpha$")
    fig2.tight_layout()
    fig2.savefig(out_traj, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out_traj.resolve()}")


def plot_dashboard_lines(path: Path, out: Path, dpi: int, show_bars: bool) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    a, M, row_labels, p_not_y0, p_sarc_y0, p_not_y1, p_sarc_y1, pred_rate, mean_p, acc = load_arrays(data)
    n_a = len(a)
    idx0 = int(np.argmin(np.abs(a)))
    a0 = float(a[idx0])
    baseline = M[:, idx0 : idx0 + 1]
    Delta = M - baseline

    i_best = int(np.nanargmax(acc)) if np.any(np.isfinite(acc)) else idx0
    a_best = float(a[i_best])
    acc_best = float(acc[i_best])

    nrows = 4 if show_bars else 3
    height_ratios = ([1.0, 1.0, 1.0, 1.15] if show_bars else [1.0, 1.0, 1.1])
    fig = plt.figure(figsize=(12, 10 if show_bars else 8.2), dpi=dpi)
    gs = GridSpec(nrows, 2, figure=fig, height_ratios=height_ratios, hspace=0.32, wspace=0.22)

    def mark_alpha(ax) -> None:
        ax.axvline(a_best, color="#9467bd", linestyle=":", linewidth=1.3, alpha=0.9, label=rf"max acc $\alpha$={a_best:g}")
        ax.set_xlabel(r"$\alpha$")

    # --- Row 1: overall rates + accuracy
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(a, pred_rate, "o-", color="#1f77b4", label="pred_sarcasm_rate", markersize=4)
    ax1.plot(a, mean_p, "s--", color="#ff7f0e", label="mean_prob_sarcastic", markersize=3.5)
    ax1.plot(a, acc, "^-", color="#d62728", label="accuracy", markersize=4)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("value in [0, 1]")
    ax1.set_title("Overall metrics vs steering strength")
    mark_alpha(ax1)
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.25)
    ax1.text(
        0.99,
        0.03,
        f"best accuracy: {acc_best:.3f} @ α={a_best:g}",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    # --- Row 2: stratified P(pred sarc | true class)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(a, M[2], "o-", color="#4c72b0", label="P(pred sarc | true ¬sarc)", markersize=4)
    ax2.plot(a, M[3], "s-", color="#dd8452", label="P(pred sarc | true sarc)", markersize=4)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("fraction predicting sarcastic")
    ax2.set_title("Who gets pushed toward “sarcastic”? (within each true class)")
    mark_alpha(ax2)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.25)

    # --- Row 3: Δ vs baseline (lines)
    ax3 = fig.add_subplot(gs[2, :])
    colors = ["#1f77b4", "#ff7f0e", "#4c72b0", "#dd8452", "#d62728"]
    for i in range(5):
        ax3.plot(a, Delta[i], "o-", color=colors[i], label=row_labels[i], markersize=3, linewidth=1.1)
    ax3.axhline(0.0, color="gray", linestyle="-", linewidth=0.9, alpha=0.5)
    ax3.set_ylabel(r"$\Delta$ vs baseline")
    ax3.set_title(f"Change from baseline (α = {a0:g})")
    mark_alpha(ax3)
    ax3.legend(loc="best", fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.25)

    # --- Optional row 4: stacked bars
    if show_bars:
        x = np.arange(n_a)
        w = 0.72
        ax_s0 = fig.add_subplot(gs[3, 0])
        ax_s0.bar(x, p_not_y0, width=w, label="pred ¬sarcastic", color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax_s0.bar(x, p_sarc_y0, width=w, bottom=p_not_y0, label="pred sarcastic", color="#DD8452", edgecolor="white", linewidth=0.3)
        ax_s0.set_xticks(x)
        ax_s0.set_xticklabels([f"{v:g}" for v in a], rotation=45, ha="right", fontsize=7)
        ax_s0.set_ylim(0, 1)
        ax_s0.set_ylabel("fraction of true ¬sarcastic")
        ax_s0.set_xlabel(r"$\alpha$ (index)")
        ax_s0.set_title("Prediction mix | true ¬sarcastic")
        ax_s0.legend(loc="upper right", fontsize=7)
        ax_s0.axvline(idx0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax_s1 = fig.add_subplot(gs[3, 1])
        ax_s1.bar(x, p_not_y1, width=w, label="pred ¬sarcastic", color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax_s1.bar(x, p_sarc_y1, width=w, bottom=p_not_y1, label="pred sarcastic", color="#DD8452", edgecolor="white", linewidth=0.3)
        ax_s1.set_xticks(x)
        ax_s1.set_xticklabels([f"{v:g}" for v in a], rotation=45, ha="right", fontsize=7)
        ax_s1.set_ylim(0, 1)
        ax_s1.set_ylabel("fraction of true sarcastic")
        ax_s1.set_xlabel(r"$\alpha$ (index)")
        ax_s1.set_title("Prediction mix | true sarcastic")
        ax_s1.legend(loc="upper right", fontsize=7)
        ax_s1.axvline(idx0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ckpt = checkpoint_display_name(str(data.get("checkpoint", "")))
    fig.suptitle(f"Steering dashboard — {ckpt}, layer {data.get('layer', '?')}", fontsize=13, y=1.01)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.resolve()}")

    _plot_trajectory_figure(out, dpi, a, pred_rate, acc, n_a)


def plot_dashboard_heatmap(path: Path, out: Path, dpi: int) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    a, M, row_labels, p_not_y0, p_sarc_y0, p_not_y1, p_sarc_y1, pred_rate, _, acc = load_arrays(data)
    n_a = len(a)

    idx0 = int(np.argmin(np.abs(a)))
    baseline = M[:, idx0 : idx0 + 1]
    Delta = M - baseline

    fig = plt.figure(figsize=(14, 11), dpi=dpi)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.35], hspace=0.35, wspace=0.28)

    # --- A: absolute heatmap
    ax_h = fig.add_subplot(gs[0, :])
    im = ax_h.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_h.set_yticks(np.arange(5))
    ax_h.set_yticklabels(row_labels, fontsize=9)
    ax_h.set_xticks(np.arange(n_a))
    ax_h.set_xticklabels([f"{x:g}" for x in a], rotation=45, ha="right", fontsize=8)
    ax_h.set_xlabel(r"$\alpha$")
    ax_h.set_title("A) Metrics × α (absolute)")
    plt.colorbar(im, ax=ax_h, fraction=0.02, pad=0.02, label="value")

    # annotate if not too many columns
    if n_a <= 15:
        for i in range(5):
            for j in range(n_a):
                val = M[i, j]
                if not np.isfinite(val):
                    continue
                ax_h.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="black")

    # --- B: delta heatmap
    ax_d = fig.add_subplot(gs[1, :])
    dmax = float(np.nanmax(np.abs(Delta)))
    if dmax < 1e-9:
        dmax = 1e-9
    norm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    im2 = ax_d.imshow(Delta, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax_d.set_yticks(np.arange(5))
    ax_d.set_yticklabels(row_labels, fontsize=9)
    ax_d.set_xticks(np.arange(n_a))
    ax_d.set_xticklabels([f"{x:g}" for x in a], rotation=45, ha="right", fontsize=8)
    ax_d.set_xlabel(r"$\alpha$")
    ax_d.axvline(idx0, color="lime", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_d.set_title(f"B) Δ vs α={a[idx0]:g} (baseline column)")
    plt.colorbar(im2, ax=ax_d, fraction=0.02, pad=0.02, label="Δ")

    if n_a <= 15:
        for i in range(5):
            for j in range(n_a):
                val = Delta[i, j]
                if not np.isfinite(val):
                    continue
                ax_d.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=6, color="black")

    # --- C: stacked bars true y=0
    ax_s0 = fig.add_subplot(gs[2, 0])
    x = np.arange(n_a)
    w = 0.72
    ax_s0.bar(x, p_not_y0, width=w, label="pred ¬sarcastic", color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax_s0.bar(x, p_sarc_y0, width=w, bottom=p_not_y0, label="pred sarcastic", color="#DD8452", edgecolor="white", linewidth=0.3)
    ax_s0.set_xticks(x)
    ax_s0.set_xticklabels([f"{v:g}" for v in a], rotation=45, ha="right", fontsize=7)
    ax_s0.set_ylim(0, 1)
    ax_s0.set_ylabel("fraction of true ¬sarcastic")
    ax_s0.set_xlabel(r"$\alpha$")
    ax_s0.set_title("C) Prediction mix | true label = ¬sarcastic (y=0)")
    ax_s0.legend(loc="upper right", fontsize=8)
    ax_s0.axvline(idx0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # --- D: stacked bars true y=1
    ax_s1 = fig.add_subplot(gs[2, 1])
    ax_s1.bar(x, p_not_y1, width=w, label="pred ¬sarcastic", color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax_s1.bar(x, p_sarc_y1, width=w, bottom=p_not_y1, label="pred sarcastic", color="#DD8452", edgecolor="white", linewidth=0.3)
    ax_s1.set_xticks(x)
    ax_s1.set_xticklabels([f"{v:g}" for v in a], rotation=45, ha="right", fontsize=7)
    ax_s1.set_ylim(0, 1)
    ax_s1.set_ylabel("fraction of true sarcastic")
    ax_s1.set_xlabel(r"$\alpha$")
    ax_s1.set_title("D) Prediction mix | true label = sarcastic (y=1)")
    ax_s1.legend(loc="upper right", fontsize=8)
    ax_s1.axvline(idx0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ckpt = checkpoint_display_name(str(data.get("checkpoint", "")))
    fig.suptitle(f"Steering dashboard — {ckpt}, layer {data.get('layer', '?')}", fontsize=13, y=1.01)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.resolve()}")

    _plot_trajectory_figure(out, dpi, a, pred_rate, acc, n_a)


def plot_dashboard(path: Path, out: Path, dpi: int, layout: str, show_bars: bool) -> None:
    if layout == "heatmap":
        plot_dashboard_heatmap(path, out, dpi)
    else:
        plot_dashboard_lines(path, out, dpi, show_bars=show_bars)


def plot_lines_classic(path: Path, out: Path, dpi: int) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    a, M, *_rest = load_arrays(data)
    pred_rate = M[0]
    p0, p1 = M[2], M[3]
    acc = M[4]

    ckpt = checkpoint_display_name(str(data.get("checkpoint", "")))
    layer = data.get("layer", "?")
    rows_meta = data.get("test_metrics_by_alpha") or []
    n_test = int(rows_meta[0]["n"]) if rows_meta and rows_meta[0].get("n") is not None else None

    alpha_tick_labels = [f"{x:g}" for x in a]

    fig = plt.figure(figsize=(19.5, 6.9), dpi=dpi)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.0, 0.17, 0.15],
        width_ratios=[1, 1, 1],
        hspace=0.52,
        wspace=0.34,
    )

    ax_l = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1], sharex=ax_l)
    ax_r = fig.add_subplot(gs[0, 2], sharex=ax_l)

    def style_axes(ax) -> None:
        ax.grid(True, alpha=0.32, linestyle="-", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        ax.set_xticks(a)
        ax.set_xticklabels(alpha_tick_labels, rotation=45, ha="right", fontsize=7.5)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_xlabel(r"$\alpha$", fontsize=13, labelpad=8)

    ax_l.plot(a, pred_rate, "o-", color="#4C72B0", markersize=4, label="pred_sarcasm_rate (argmax = sarcasm)")
    ax_l.set_ylabel("fraction", fontsize=13)
    ax_l.set_ylim(-0.05, 1.05)
    ax_l.set_title(
        "Aggregate on test set\nfraction predicted sarcastic (argmax)",
        fontsize=13,
        color="#1a1a1a",
    )
    style_axes(ax_l)

    ax_m.plot(a, p0, "o-", color="#8172B3", markersize=4, label="gold ¬sarcastic (y=0): P(pred sarc | y=0)")
    ax_m.plot(a, p1, "s-", color="#CCB974", markersize=4, label="gold sarcastic (y=1): P(pred sarc | y=1)")
    ax_m.set_ylabel("conditional probability", fontsize=13)
    ax_m.set_ylim(-0.05, 1.05)
    ax_m.set_title(
        "Stratified by true label\nfalse-positive rate vs. sarcasm recall",
        fontsize=13,
        color="#1a1a1a",
    )
    style_axes(ax_m)

    ax_r.plot(a, acc, "o-", color="#C44E52", markersize=4, label="accuracy")
    ax_r.set_ylabel("accuracy", fontsize=13)
    ax_r.set_ylim(-0.05, 1.05)
    ax_r.set_title(
        "End-to-end correctness\nfraction of samples with correct argmax",
        fontsize=13,
        color="#1a1a1a",
    )
    style_axes(ax_r)

    h_l, lab_l = ax_l.get_legend_handles_labels()
    h_m, lab_m = ax_m.get_legend_handles_labels()
    h_r, lab_r = ax_r.get_legend_handles_labels()
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    ax_leg.legend(
        h_l + h_m + h_r,
        lab_l + lab_m + lab_r,
        loc="center",
        ncol=4,
        fontsize=13,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.95,
    )

    ax_note = fig.add_subplot(gs[2, :])
    ax_note.axis("off")
    note = f"checkpoint: {ckpt}   ·   steering layer: {layer}"
    if n_test is not None:
        note += f"   ·   test size (per α): {n_test}"
    ax_note.text(0.5, 0.55, note, ha="center", va="center", fontsize=12, color="#444444")

    fig.suptitle(
        "Steering sweep along v̂ (train class-mean difference, L2-normalized)",
        fontsize=15,
        y=0.98,
        color="#111111",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.05, hspace=0.64, wspace=0.36)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


def main() -> None:
    args = parse_args()
    path = Path(args.input_json)
    out = Path(args.output) if args.output else path.parent / "steering_dashboard.png"
    plot_dashboard(path, out, args.dpi, layout=args.layout, show_bars=args.bars)
    if args.lines:
        plot_lines_classic(path, path.parent / "steering_curves.png", args.dpi)


if __name__ == "__main__":
    main()
