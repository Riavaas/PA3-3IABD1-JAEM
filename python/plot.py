#!/usr/bin/env python3
"""Visualisation des datasets (linear, xor)."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def plot_linear() -> None:
    import matplotlib.pyplot as plt
    path = DATASETS / "linear.csv"
    if not path.exists():
        print("Générer d'abord: python generate_datasets.py")
        return
    rows = load_csv(path)
    x1 = [float(r["x1"]) for r in rows]
    x2 = [float(r["x2"]) for r in rows]
    y = [int(r["y"]) for r in rows]
    plt.scatter(x1, x2, c=y, cmap="viridis")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear dataset")
    plt.savefig(ROOT / "datasets" / "linear_plot.png")
    plt.close()
    print("Saved datasets/linear_plot.png")


def plot_xor() -> None:
    import matplotlib.pyplot as plt
    path = DATASETS / "xor.csv"
    if not path.exists():
        print("Générer d'abord: python generate_datasets.py")
        return
    rows = load_csv(path)
    x1 = [float(r["x1"]) for r in rows]
    x2 = [float(r["x2"]) for r in rows]
    y = [int(r["y"]) for r in rows]
    plt.scatter(x1, x2, c=y, cmap="viridis")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("XOR dataset")
    plt.savefig(ROOT / "datasets" / "xor_plot.png")
    plt.close()
    print("Saved datasets/xor_plot.png")


if __name__ == "__main__":
    plot_linear()
    plot_xor()
