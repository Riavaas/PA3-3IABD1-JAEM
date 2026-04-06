import numpy as np
import csv
import os

np.random.seed(42)

n = 200
X = np.random.randn(n, 2) * 2
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int).tolist()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xor.csv")

with open(out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x1", "x2", "label"])
    for i in range(n):
        writer.writerow([round(X[i, 0], 6), round(X[i, 1], 6), y[i]])

print(f"xor.csv généré ({n} points)")
