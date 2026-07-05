#!/usr/bin/env python3
"""
Affiche des exemples depuis datasets/transformed/nb/(normalisee|non_normalisee).

Chaque ligne de X.npy = une image NB aplatie (H*W).
On reconstruit l'image et on affiche les 3 premières.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS_INV = {0: "dry_road", 1: "wet_road", 2: "snowy_road"}


def infer_hw_from_d_gray(d: int) -> tuple[int, int]:
    h = int(math.isqrt(d))
    if h * h != d:
        raise ValueError(f"Impossible d'inférer H=W depuis d={d} (pas un carré parfait)")
    return h, h


def main() -> None:
    here = Path(__file__).resolve()
    project_root = here.parents[1]

    parser = argparse.ArgumentParser(description="Visualise 3 images NB depuis datasets/transformed.")
    parser.add_argument("--normalized", action="store_true", help="Utilise le dossier normalisee (sinon non_normalisee)")
    parser.add_argument("--count", type=int, default=3, help="Nombre d'images à afficher (défaut: 3)")
    args = parser.parse_args()

    sub = "normalisee" if args.normalized else "non_normalisee"
    base = project_root / "datasets" / "transformed" / "nb" / sub

    X = np.load(base / "X.npy")
    y = np.load(base / "y.npy")

    n, d = X.shape
    h, w = infer_hw_from_d_gray(d)
    k = min(int(args.count), n)

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes = [axes]

    for i in range(k):
        img = X[i].reshape(h, w)
        if not args.normalized:
            img = img.astype(np.uint8)
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1 if args.normalized else 255)
        axes[i].axis("off")
        axes[i].set_title(f"i={i}, y={int(y[i])} ({LABELS_INV.get(int(y[i]), '?')})")

    fig.suptitle(f"NB/{sub} - {k} exemple(s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

