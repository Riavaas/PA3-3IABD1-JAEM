import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linear_model import LinearModel
from models.mlp import MLP

np.random.seed(42)

splits = "datasets/splits/local"
X_train = np.load(f"{splits}/X_train.npy")
X_test  = np.load(f"{splits}/X_test.npy")
y_train = np.load(f"{splits}/y_train.npy")
y_test  = np.load(f"{splits}/y_test.npy")

CLASSES = ["hello_kitty", "sanrio_other", "other"]


def binariser(y, classe):
    return (y == classe).astype(int)


fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Modèles sur dataset réel (images 64x64)")

resultats = {}

for idx, (nom_classe, label) in enumerate(zip(CLASSES, [0, 1, 2])):
    y_tr = binariser(y_train, label)
    y_te = binariser(y_test, label)

    lin = LinearModel(lr=0.01, epochs=300)
    lin.fit(X_train, y_tr)
    acc_lin_train = lin.score(X_train, y_tr)
    acc_lin_test  = lin.score(X_test, y_te)

    mlp = MLP(hidden_sizes=[64, 32], lr=0.05, epochs=300)
    mlp.fit(X_train, y_tr)
    acc_mlp_train = mlp.score(X_train, y_tr)
    acc_mlp_test  = mlp.score(X_test, y_te)

    resultats[nom_classe] = {
        "lin_train": acc_lin_train, "lin_test": acc_lin_test,
        "mlp_train": acc_mlp_train, "mlp_test": acc_mlp_test,
    }

    axes[0][idx].plot(lin.losses, label="linéaire")
    axes[0][idx].set_title(f"Loss — {nom_classe}")
    axes[0][idx].set_xlabel("époque")
    axes[0][idx].set_ylabel("loss")
    axes[0][idx].legend()

    axes[1][idx].plot(mlp.losses, label="MLP", color="orange")
    axes[1][idx].set_title(f"Loss MLP — {nom_classe}")
    axes[1][idx].set_xlabel("époque")
    axes[1][idx].set_ylabel("loss")
    axes[1][idx].legend()

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats_images.png")
plt.savefig(out, dpi=150)
plt.show()

print(f"\n{'classe':<15} {'lin train':>10} {'lin test':>10} {'mlp train':>10} {'mlp test':>10}")
print("-" * 60)
for cls, r in resultats.items():
    print(f"{cls:<15} {r['lin_train']:>10.3f} {r['lin_test']:>10.3f} {r['mlp_train']:>10.3f} {r['mlp_test']:>10.3f}")

print(f"\nGraphe sauvegardé : {out}")
