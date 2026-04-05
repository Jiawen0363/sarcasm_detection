"""
Plot cosine_train_diff_vs_test_means.json: per-layer cos(v, mu_test_y0), cos(v, mu_test_y1),
and cos(mu_te0, mu_te1). Single panel, no title, legend below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_json",
        type=str,
        default=str(base / "outputs" / "roberta_frozen_encoder" / "cosine_train_diff_vs_test_means.json"),
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Default: <json_stem>.png next to JSON",
    )
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_json)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = sorted(data["layers"], key=lambda x: x["layer"])
    x = np.array([L["layer"] for L in layers])
    c0 = np.array([L["cos_v_mu_test_class0"] for L in layers])
    c1 = np.array([L["cos_v_mu_test_class1"] for L in layers])
    c01 = np.array([L["cos_mu_test_class0_mu_test_class1"] for L in layers])

    out = Path(args.output) if args.output else path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=args.dpi, layout="constrained")

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.plot(x, c0, "o-", color="#4C72B0", label=r"$\cos(v,\ \mu_{\mathrm{test}}^{(0)})$ not sarcastic", markersize=6)
    ax.plot(x, c1, "s-", color="#DD8452", label=r"$\cos(v,\ \mu_{\mathrm{test}}^{(1)})$ sarcastic", markersize=6)
    ax.plot(
        x,
        c01,
        ":",
        color="#555555",
        linewidth=1.5,
        alpha=0.85,
        label=r"$\cos(\mu_{\mathrm{test}}^{(0)},\mu_{\mathrm{test}}^{(1)})$",
    )
    ax.set_xlabel("Encoder layer index")
    ax.set_ylabel("cosine")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=8,
        frameon=True,
    )

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
