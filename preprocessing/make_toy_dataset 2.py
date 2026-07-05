import os

# Ce script fabrique 2 datasets de test (des points)

# Dossier ou on range les fichiers. On le cree s'il n'existe pas encore.
DOSSIER = "../datasets/toy"
os.makedirs(DOSSIER, exist_ok=True)


# 1) Dataset lineaire : deux groupes de points separables par une droite.
#    Chaque point = (x1, x2, label). label 0 = groupe du bas, label 1 = groupe du haut.
points_lineaire = [
    (1, 1, 0),
    (2, 2, 0),
    (3, 3, 0),
    (6, 6, 1),
    (7, 7, 1),
    (8, 8, 1),
]

# 2) Dataset XOR : 4 points en croix, qu'aucune droite ne peut separer.
#    label 1 quand x1 et x2 sont differents, label 0 quand ils sont pareils.
points_xor = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]


def ecrire_csv(chemin, points):
    """Ecrit la liste de points dans un fichier CSV (une ligne = un point)."""
    with open(chemin, "w") as f:
        f.write("x1,x2,label\n")          # la premiere ligne donne le nom des colonnes
        for (x1, x2, label) in points:    # on parcourt chaque point
            f.write(f"{x1},{x2},{label}\n")
    print("Ecrit :", chemin, "(", len(points), "points )")


# On genere les deux fichiers.
ecrire_csv(os.path.join(DOSSIER, "linear.csv"), points_lineaire)
ecrire_csv(os.path.join(DOSSIER, "xor.csv"), points_xor)

print("Termine.")