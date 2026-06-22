import sys
import numpy as np
import matplotlib.pyplot as plt

# Ce script DESSINE seulement : il lit les points (CSV) et les poids
# de la droite (calcules par le programme C), puis trace le graphe.
# Il n'entraine aucun modele.

# 1) Quels fichiers lire ? (donnes en argument, sinon valeurs par defaut)
chemin_csv = sys.argv[1] if len(sys.argv) >= 2 else "datasets/toy/linear.csv"
chemin_poids = sys.argv[2] if len(sys.argv) >= 3 else "models/poids.txt"

# 2) Lecture des points du CSV.
#    On saute la 1ere ligne (l'en-tete x1,x2,label) avec skiprows=1.
#    delimiter="," car les colonnes sont separees par des virgules.
donnees = np.loadtxt(chemin_csv, delimiter=",", skiprows=1)
x1 = donnees[:, 0]      # 1ere colonne
x2 = donnees[:, 1]      # 2e colonne
label = donnees[:, 2]   # 3e colonne (la classe)

# 3) Lecture des poids w1, w2, b ecrits par le programme C.
w1, w2, b = np.loadtxt(chemin_poids)
print("Poids lus :", "w1 =", w1, " w2 =", w2, " b =", b)

# 4) On dessine les points, une couleur par classe.
plt.figure()
plt.scatter(x1[label == 0], x2[label == 0], color="royalblue", label="classe 0")
plt.scatter(x1[label == 1], x2[label == 1], color="orangered", label="classe 1")

# 5) On dessine la droite de decision : w1*x1 + w2*x2 + b = 0.
#    On choisit une plage de x1 un peu plus large que les points.
xmin, xmax = x1.min() - 1, x1.max() + 1

if w2 != 0:
    # cas general : on isole x2 -> x2 = -(w1*x1 + b) / w2
    xs = np.array([xmin, xmax])
    ys = -(w1 * xs + b) / w2
    plt.plot(xs, ys, color="black", label="droite de decision")
elif w1 != 0:
    # cas particulier : droite verticale x1 = -b / w1
    plt.axvline(x=-b / w1, color="black", label="droite de decision")
# (si w1 et w2 valent 0, il n'y a pas de droite a tracer)

# 6) Mise en forme et sauvegarde.
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Modele lineaire : points + droite de decision")
plt.savefig("plot_linear.png", dpi=120)   # sauvegarde l'image (pour le rapport)
print("Image sauvegardee : plot_linear.png")
plt.show()   # ouvre une fenetre (si tu es en local)