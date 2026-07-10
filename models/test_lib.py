import ctypes

# On charge notre bibliotheque C compilee
lib = ctypes.CDLL("./linear_model.so")

print("Bibliotheque chargee :", lib)

import numpy as np

# Des raccourcis pour dire "pointeur vers des floats" et "pointeur vers des ints"
c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p = ctypes.POINTER(ctypes.c_int)

# On decrit les arguments de fit, dans le meme ordre que dans le C :
# (X, y, n, d, epochs, lr, W, b)
lib.fit.argtypes = [c_float_p, c_int_p, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_float, c_float_p, c_float_p]

# Et ceux de predict : (X, n, d, W, b, out)
lib.predict.argtypes = [c_float_p, ctypes.c_int, ctypes.c_int,
                        c_float_p, c_float_p, c_int_p]

print("Types declares, fit et predict sont prets")

# Petit jeu de test a 3 classes, 2 features, bien separees
X = np.array([
    [0, 0], [1, 0], [0, 1],        # classe 0 (coin bas gauche)
    [10, 10], [11, 10], [10, 11],  # classe 1 (loin en haut a droite)
    [0, 10], [1, 10], [0, 11],     # classe 2 (haut gauche)
], dtype=np.float32)
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)

n, d = X.shape   # n = nombre de points, d = nombre de features
K = 3            # 3 classes

# Python prepare les tableaux de sortie, le C les remplira
W = np.zeros(K * d, dtype=np.float32)
b = np.zeros(K, dtype=np.float32)

# Deux petits raccourcis pour donner l'adresse memoire d'un tableau au C
def fp(a): return a.ctypes.data_as(c_float_p)
def ip(a): return a.ctypes.data_as(c_int_p)

# On entraine : 50 epochs, lr = 0.1
lib.fit(fp(X), ip(y), n, d, 50, ctypes.c_float(0.1), fp(W), fp(b))

# On predit sur les memes points
out = np.zeros(n, dtype=np.int32)
lib.predict(fp(X), n, d, fp(W), fp(b), ip(out))

print("vraies classes :", y)
print("predictions    :", out)
print("accuracy       :", (out == y).mean())