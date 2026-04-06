import numpy as np


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y


def normaliser(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std
