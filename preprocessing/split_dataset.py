import numpy as np
from pathlib import Path
import argparse


def split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(y))
    cut = int(len(y) * (1 - test_size))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def split_and_save(input_dir, output_dir, test_size=0.2, seed=42):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(input_dir / "X.npy")
    y = np.load(input_dir / "y.npy")

    X_train, X_test, y_train, y_test = split(X, y, test_size, seed)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    print(f"Train : {len(y_train)} | Test : {len(y_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_and_save(args.input, args.output, args.test_size, args.seed)
