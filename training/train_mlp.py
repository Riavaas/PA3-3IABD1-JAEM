import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP
from preprocessing.utils import load_csv

np.random.seed(42)

datasets = {
    "linear": "datasets/linear.csv",
    "xor": "datasets/xor.csv",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("MLP")

for idx, (nom, path) in enumerate(datasets.items()):
    X, y = load_csv(path)
    model = MLP(hidden_sizes=[8], lr=0.5, epochs=3000)
    model.fit(X, y)
    acc = model.score(X, y)

    ax_boundary = axes[idx][0]
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300), np.linspace(x2_min, x2_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax_boundary.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax_boundary.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=25)
    ax_boundary.set_title(f"{nom} — accuracy : {acc:.2f}")
    ax_boundary.set_xlabel("x1")
    ax_boundary.set_ylabel("x2")

    axes[idx][1].plot(model.losses)
    axes[idx][1].set_title(f"Loss — {nom}")
    axes[idx][1].set_xlabel("époque")
    axes[idx][1].set_ylabel("loss")

    print(f"{nom} → accuracy = {acc:.4f}")

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats_mlp.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Graphe sauvegardé : {out}")
