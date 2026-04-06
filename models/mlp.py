import numpy as np


class MLP:
    def __init__(self, hidden_sizes=None, lr=0.01, epochs=1000):
        if hidden_sizes is None:
            hidden_sizes = [8]
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.losses = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_deriv(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        n, d = X.shape
        sizes = [d] + self.hidden_sizes + [1]

        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros(sizes[i + 1]))

        self.losses = []
        y = y.reshape(-1, 1).astype(float)

        for _ in range(self.epochs):
            activations = [X]
            a = X
            for w, b in zip(self.weights, self.biases):
                a = self._sigmoid(a @ w + b)
                activations.append(a)

            out = activations[-1]
            loss = -np.mean(y * np.log(out + 1e-9) + (1 - y) * np.log(1 - out + 1e-9))
            self.losses.append(loss)

            delta = out - y
            for i in reversed(range(len(self.weights))):
                dw = activations[i].T @ delta / n
                db = np.mean(delta, axis=0)
                if i > 0:
                    delta = (delta @ self.weights[i].T) * self._sigmoid_deriv(activations[i])
                self.weights[i] -= self.lr * dw
                self.biases[i] -= self.lr * db

        return self

    def predict(self, X):
        a = X
        for w, b in zip(self.weights, self.biases):
            a = self._sigmoid(a @ w + b)
        return (a.reshape(-1) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
