#!/usr/bin/env python3
"""
Export minimal pour lire facilement le dataset en C.

Source:
  datasets/transformed/nb/normalisee/{X.npy,y.npy}

Sortie (binaire, little-endian):
  datasets/for_c/nb_normalisee_X.f32bin
    - int32 n
    - int32 d
    - float32 data[n*d] (row-major)
  datasets/for_c/nb_normalisee_y.i32bin
    - int32 n
    - int32 labels[n]

Pourquoi ce format ?
- `.npy` est pratique en Python, mais plus pénible à parser en C.
- Ici on choisit un en-tête ultra-simple + un bloc contigu de valeurs pour
  pouvoir faire un seul `fread` côté C.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    here = Path(__file__).resolve()
    # Racine du projet = dossier parent de preprocessing/
    project_root = here.parents[1]

    parser = argparse.ArgumentParser(description="Exporte nb/normalisee en binaire simple pour C.")
    parser.add_argument(
        "--input-dir",
        default=str(project_root / "datasets" / "transformed" / "nb" / "normalisee"),
        help="Dossier contenant X.npy et y.npy",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "datasets" / "for_c"),
        help="Dossier de sortie",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    # Crée le dossier de sortie si besoin.
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / "X.npy"
    y_path = input_dir / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Introuvable: {x_path} ou {y_path}. Lance build_dataset.py avant.")

    # Dtypes stables pour le C: float32 (X) et int32 (y).
    X = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.int32, copy=False)

    if X.ndim != 2:
        raise ValueError(f"X doit être 2D, reçu shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y doit être 1D, reçu shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Incohérent: X a {X.shape[0]} lignes mais y a {y.shape[0]} éléments")

    n, d = X.shape

    out_x = output_dir / "nb_normalisee_X.f32bin"
    out_y = output_dir / "nb_normalisee_y.i32bin"

    # X: [int32 n][int32 d][float32 n*d]
    with open(out_x, "wb") as f:
        np.array([n], dtype=np.int32).tofile(f)
        np.array([d], dtype=np.int32).tofile(f)
        # Layout contigu garanti: on écrit n*d float32 d'un bloc.
        np.ascontiguousarray(X).tofile(f)

    # y: [int32 n][int32 n labels]
    with open(out_y, "wb") as f:
        np.array([n], dtype=np.int32).tofile(f)
        np.ascontiguousarray(y).tofile(f)

    print("Export terminé.")
    print(f"  - X: {out_x} (n={n}, d={d}, float32)")
    print(f"  - y: {out_y} (n={n}, int32)")


if __name__ == "__main__":
    main()

