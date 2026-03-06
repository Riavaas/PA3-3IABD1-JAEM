#!/usr/bin/env python3
"""Génère les jeux de données linear.csv et xor.csv dans datasets/."""

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


def generate_linear(n: int = 50, output: Path = DATASETS / "linear.csv") -> None:
    """Données linéairement séparables : y = 2*x1 + 1 + bruit."""
    DATASETS.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x1", "x2", "y"])
        for i in range(n):
            x1 = i / n
            x2 = 2 * x1 + 1 + (i % 5) * 0.1
            y = 1 if x2 > 2 * x1 + 1 else 0
            w.writerow([f"{x1:.4f}", f"{x2:.4f}", y])
    print(f"Écrit {output}")


def generate_xor(n: int = 100, output: Path = DATASETS / "xor.csv") -> None:
    """Données XOR : 4 clusters."""
    DATASETS.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x1", "x2", "y"])
        for _ in range(n):
            b1 = random.randint(0, 1)
            b2 = random.randint(0, 1)
            x1 = b1 * 0.8 + 0.1 + random.gauss(0, 0.05)
            x2 = b2 * 0.8 + 0.1 + random.gauss(0, 0.05)
            y = b1 ^ b2
            w.writerow([f"{x1:.4f}", f"{x2:.4f}", y])
    print(f"Écrit {output}")


if __name__ == "__main__":
    random.seed(42)
    generate_linear()
    generate_xor()
