import numpy as np
import csv
import os

np.random.seed(42)

n = 200
X0 = np.random.randn(n // 2, 2) + np.array([2.5, 2.5])
X1 = np.random.randn(n // 2, 2) + np.array([-2.5, -2.5])
X = np.vstack([X0, X1])
y = [0] * (n // 2) + [1] * (n // 2)

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear.csv")

with open(out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x1", "x2", "label"])
    for i in range(n):
        writer.writerow([round(X[i, 0], 6), round(X[i, 1], 6), y[i]])

print(f"linear.csv généré ({n} points)")
