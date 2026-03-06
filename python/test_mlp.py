#!/usr/bin/env python3
"""Tests du MLP (à brancher sur ml-lib ou version Python)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def test_xor_dataset_exists() -> None:
    csv_path = ROOT / "datasets" / "xor.csv"
    assert csv_path.exists(), "datasets/xor.csv manquant (lancer generate_datasets.py)"
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) >= 2


def test_mlp_placeholder() -> None:
    assert True
