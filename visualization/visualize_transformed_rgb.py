#!/usr/bin/env python3
"""
Affiche des exemples depuis datasets/transformed/rgb/(normalisee|non_normalisee).

Chaque ligne de X.npy = une image RGB aplatie (H*W*3).
On reconstruit l'image et on affiche les 3 premières.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS_INV = {0: "dry_road", 1: "wet_road", 2: "snowy_road"}


def infer_hw_from_d_rgb(d: int) -> tuple[int, int]:
    if d % 3 != 0:
        raise ValueError(f"Dimension RGB invalide: d={d} (pas multiple de 3)")
    pix = d // 3
    h = int(math.isqrt(pix))
    if h * h != pix:
        raise ValueError(f"Impossible d'inférer H=W depuis d={d} (pix={pix} n'est pas un carré parfait)")
    return h, h


def main() -> None:
    here = Path(__file__).resolve()
    project_root = here.parents[1]

    parser = argparse.ArgumentParser(description="Visualise 3 images RGB depuis datasets/transformed.")
    parser.add_argument("--normalized", action="store_true", help="Utilise le dossier normalisee (sinon non_normalisee)")
    parser.add_argument("--count", type=int, default=3, help="Nombre d'images à afficher (défaut: 3)")
    args = parser.parse_args()

    sub = "normalisee" if args.normalized else "non_normalisee"
    base = project_root / "datasets" / "transformed" / "rgb" / sub

    X = np.load(base / "X.npy")
    y = np.load(base / "y.npy")

    n, d = X.shape
    h, w = infer_hw_from_d_rgb(d)
    k = min(int(args.count), n)

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes = [axes]

    for i in range(k):
        img = X[i].reshape(h, w, 3)
        if not args.normalized:
            img = img.astype(np.uint8)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"i={i}, y={int(y[i])} ({LABELS_INV.get(int(y[i]), '?')})")

    fig.suptitle(f"RGB/{sub} - {k} exemple(s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

