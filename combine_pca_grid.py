"""
Combine layer_00.png ... layer_11.png into one 3x4 figure (row-major: layers 0–3 top row).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="3x4 grid of PCA layer PNGs")
    p.add_argument(
        "--input_dir",
        type=str,
        default=str(base / "outputs" / "roberta_split" / "pca_first_token"),
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output path (.png / .pdf). Default: <input_dir>/layers_grid_3x4.png",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--fig_title",
        type=str,
        default="",
        help="Optional suptitle above the grid",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    indir = Path(args.input_dir)
    out_path = Path(args.output) if args.output else indir / "layers_grid_3x4.png"

    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), dpi=args.dpi)
    axes_flat = axes.flatten()

    for i in range(12):
        path = indir / f"layer_{i:02d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        img = mpimg.imread(path)
        ax = axes_flat[i]
        ax.imshow(img)
        ax.axis("off")

    if args.fig_title:
        fig.suptitle(args.fig_title, fontsize=14, y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
