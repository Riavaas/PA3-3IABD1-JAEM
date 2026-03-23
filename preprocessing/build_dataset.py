##DEBUT

#    definir le chemin du dataset
#    definir la liste des classes :
#        - hello_kitty
#        - fake_hello_kitty
#        - other

#    creer une liste X pour stocker les vecteurs d’images
#    creer une liste y pour stocker les labels

#    pour chaque classe dans la liste des classes :
#        recuperer le dossier correspondant

#        pour chaque image dans ce dossier :
#            lire l’image
#            verifier qu’elle est valide

#    convertir X en matrice
#    convertir y en vecteur

#    sauvegarder X et y
#    afficher :
#        - nombre total d’images
#        - taille d’un vecteur
#        - nombre d’images par classe

#FIN


#https://www.youtube.com/watch?v=HvKpzGadlB4

import matplotlib.pyplot as plt
import numpy as np
import imageio 


def moyenne(L):
    """Retourne la moyenne des éléments de la liste L(non vide)"""
    assert len(L) > 0, "La liste L est vide"
    S = 0
    for i in range(len(L)):
        S += L[i]
    return S / len(L)

#ici la matrice X est une liste de listes, ex : L=[[1,2,3],[4,5,6],[7,8,9]]
#L[0] = [1,2,3]
#L[1] = [4,5,6]
#L[2] = [7,8,9]
#L[1][1] = 5
#len(L) = 3 = nombre de lignes, car 3 listes
#len(L[0]) = 3 = nombre de colonnes

#M = [0 for j in range(3)] sert à créer une liste de 3 zéros

#Z = [(0 for j in range(3))] for i in range(2) sert à créer une liste de 2 listes de 3 zéros

Img = imageio.imread("test.jpeg").tolist()
print(len(Img))
print(len(Img[0]))
print(Img[3][5])
plt.figure()
plt.imshow(Img)
plt.title("Image de test")
plt.show()


def conversionNB(Img):
    """convertir une image en noir et blanc"""
    n=len(Img) # nb de lignes
    p=len(Img[0]) # nb de colonnes
    for i in range(n):
        for j in range(p):
            Img[i][j] = moyenne(Img[i][j])


conversionNB(Img)
plt.figure()
plt.imshow(Img, cmap='gray')
plt.title("Image de test en noir et blanc")
plt.show()


def negatif(Img):
    """convertir une image en négatif"""
    n=len(Img) # nb de lignes
    p=len(Img[0]) # nb de colonnes
    Z=[[0 for j in range(p)] for i in range(n)] #Z la matrice nulle de même taille que Img
    for i in range(n):
        for j in range(p):
            Z[i][j] = 255 - Img[i][j]
    return Z


Z=negatif(Img)
plt.figure()
plt.imshow(Z, cmap='gray')
plt.title("Image de test en négatif")
plt.show()


def symétrie(Img):
    """convertir une image en symétrie par rapport à l'axe vertical"""
    n=len(Img) # nb de lignes
    p=len(Img[0]) # nb de colonnes
    Z=[[0 for j in range(p)] for i in range(n)] #Z la matrice nulle de même taille que Img
    for i in range(n):
        for j in range(p):
            Z[i][j] = Img[i][p-1-j]#p-1-j car on commence à 0 et on veut aller jusqu'à p-1
    return Z

S=symétrie(Img)
plt.figure()
plt.imshow(S, cmap='gray')
plt.title("Image de test en symétrie par rapport à l'axe vertical")
plt.show()



#P[i][j] = |Img[i+1][j] - Img[i-1][j]| + |Img[i][j+1] - Img[i][j-1]|
def contours(Img):
    """convertir une image en contours"""
    n=len(Img) # nb de lignes
    p=len(Img[0]) # nb de colonnes
    P=[[0 for j in range(p)] for i in range(n)] #P la matrice nulle de même taille que Img
    for i in range(1, n-1):
        for j in range(1, p-1):
            P[i][j] = abs(Img[i+1][j] - Img[i-1][j]) + abs(Img[i][j+1] - Img[i][j-1])
    return P

P=contours(Img)
plt.figure()
plt.imshow(P, cmap='gray')
plt.title("Image de test en contours")
plt.show()


P=negatif(contours(Img))
plt.figure()
plt.imshow(P, cmap='gray')
plt.title("Image de test en contours en négatif")
plt.show()

