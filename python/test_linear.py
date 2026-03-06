#!/usr/bin/env python3
"""Tests du modèle linéaire (à brancher sur ml-lib ou version Python)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Stub : charger datasets et vérifier qu'ils existent
def test_linear_dataset_exists() -> None:
    csv_path = ROOT / "datasets" / "linear.csv"
    assert csv_path.exists(), "datasets/linear.csv manquant (lancer generate_datasets.py)"
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) >= 2
    assert "x1" in lines[0] and "y" in lines[0]


def test_linear_model_import() -> None:
    # Placeholder : plus tard importer le binding C++ ou une implé Python
    assert True
