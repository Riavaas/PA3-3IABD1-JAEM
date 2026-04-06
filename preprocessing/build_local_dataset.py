import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLASSES = ["hello_kitty", "sanrio_other", "other"]
LABELS = {"hello_kitty": 0, "sanrio_other": 1, "other": 2}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SIZE = 64


def charger_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def build(sources, output_dir):
    X, y = [], []
    counts = {}

    for classe, dossier in sources.items():
        dossier = Path(dossier)
        if not dossier.exists():
            print(f"absent : {dossier}")
            continue
        n = 0
        for f in sorted(dossier.iterdir()):
            if f.suffix.lower() not in EXTENSIONS:
                continue
            try:
                img = charger_image(f)
                X.append(img.reshape(-1))
                y.append(LABELS[classe])
                n += 1
            except Exception as e:
                print(f"ignoré {f.name} : {e}")
        counts[classe] = n
        print(f"{classe} : {n} images")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", np.array(X, dtype=np.float32))
    np.save(output_dir / "y.npy", np.array(y, dtype=np.int64))

    print(f"\ntotal : {len(y)} images")
    print(f"vecteur : {len(X[0])} features")
    print(f"sauvegardé dans {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hello-kitty", required=True)
    parser.add_argument("--sanrio-other", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--output", default="datasets/transformed/local")
    args = parser.parse_args()

    sources = {
        "hello_kitty": args.hello_kitty,
        "sanrio_other": args.sanrio_other,
        "other": args.other,
    }

    build(sources, args.output)
