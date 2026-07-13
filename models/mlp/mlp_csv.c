#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * --- Validation contre les cas de tests officiels du prof ---
 * Le notebook "[Notebook] Cas de tests.ipynb" du prof definit des jeux de
 * donnees de reference (Linear Simple, Linear Multiple, Cross...) avec des
 * labels {-1,+1}. On les a regeneres en CSV {0,1} (meme convention que nos
 * fichiers) dans datasets/toy/prof_*.csv, pour verifier que ce programme se
 * comporte comme attendu sur les cas officiels, pas seulement sur les notres.
 * Resultat attendu : OK partout, y compris sur prof_cross.csv (non separable,
 * comme XOR) -- c'est justement ce que le MLP doit reussir la ou le lineaire
 * echoue.
 */

#define H 4
#define MAX_POINTS 600   /* prof_cross.csv contient 500 points */

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double calcul_erreur_sortie(double sortie, int vraie_classe) {
    return sortie - (double)vraie_classe;
}

// Corrige W1/b1 en utilisant l'ANCIEN W2 (avant sa propre correction) : chaque
// neurone cache est corrige proportionnellement a la confiance que la couche
// de sortie lui accordait au moment de l'erreur.
void corriger_cachee(double erreur_sortie, double x1, double x2, double a1[H],
                      double W1[H][2], double b1[H], double W2[H], double lr) {
    for (int h = 0; h < H; h++) {
        double da1 = erreur_sortie * W2[h];
        double dz1 = da1 * a1[h] * (1.0 - a1[h]);   // derivee de la sigmoide
        W1[h][0] -= lr * dz1 * x1;
        W1[h][1] -= lr * dz1 * x2;
        b1[h] -= lr * dz1;
    }
}

void corriger_sortie(double erreur, double a1[H], double W2[H], double *b2, double lr) {
    for (int h = 0; h < H; h++) {
        W2[h] -= lr * erreur * a1[h];
    }
    *b2 -= lr * erreur;
}

int main(int argc, char *argv[]) {
    const char *chemin = argv[1];
    char entete[100];
    double x1[MAX_POINTS], x2[MAX_POINTS];
    int label[MAX_POINTS];                   
    int n = 0;

    double W1[H][2];
    double b1[H];
    double W2[H];
    double b2;

    FILE *f = fopen(chemin, "r");
    if (!f) {
        perror(chemin);
        return 1;
    }
    printf("Fichier ouvert \n");


    fgets(entete, sizeof(entete), f); 
    while (n < MAX_POINTS && fscanf(f, "%lf,%lf,%d", &x1[n], &x2[n], &label[n]) == 3) {
        n++;
    }

    printf("Nombre de points lus : %d\n", n);

    srand(67);
    for (int h = 0; h < H; h++) {
        for (int j = 0; j < 2; j++) {
            W1[h][j] = ((double)rand() / RAND_MAX - 0.5) * 2.0; // valeurs entre -1 et 1
        }
        b1[h] = 0.0;
    }

    for (int h = 0; h < H; h++) {
        W2[h] = ((double)rand() / RAND_MAX - 0.5) * 2.0; // valeurs entre -1 et 1
    }
    b2 = 0.0;

    printf("Poids et biais initialisés \n");
    printf("W1 : \n");
    printf("[");
    for (int h = 0; h < H; h++) {
        printf("[%.2f, %.2f]", W1[h][0], W1[h][1]);
        if (h < H - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("b1 : [");
    for (int h = 0; h < H; h++) {
        printf("%.2f", b1[h]);
        if (h < H - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("W2 : [");
    for (int h = 0; h < H; h++) {
        printf("%.2f", W2[h]);
        if (h < H - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("b2 : %.2f\n", b2);


double a1[H];
double lr = 0.5;
int epochs = 2000;

// Boucle d'entrainement : couche cachee ET couche de sortie sont corrigees.
for (int epoch = 0; epoch < epochs; epoch++) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < H; h++) {
            double z = W1[h][0] * x1[i] + W1[h][1] * x2[i] + b1[h];
            a1[h] = sigmoid(z);
        }

        double z2 = b2;
        for (int h = 0; h < H; h++) {
            z2 += W2[h] * a1[h];
        }
        double sortie = sigmoid(z2);
        int prediction = (sortie > 0.5) ? 1 : 0;
        if (prediction == label[i]) correct++;

        double erreur = calcul_erreur_sortie(sortie, label[i]);
        corriger_cachee(erreur, x1[i], x2[i], a1, W1, b1, W2, lr);   // avec l'ANCIEN W2
        corriger_sortie(erreur, a1, W2, &b2, lr);                     // W2 change seulement maintenant
    }
    if ((epoch + 1) % 200 == 0 || epoch == 0) {
        printf("Epoch %d : accuracy = %.2f\n", epoch + 1, (double)correct / n);
    }
}

// Verification finale, point par point
printf("\n--- Verification finale ---\n");
int correct_final = 0;
for (int i = 0; i < n; i++) {
    for (int h = 0; h < H; h++) {
        double z = W1[h][0] * x1[i] + W1[h][1] * x2[i] + b1[h];
        a1[h] = sigmoid(z);
    }
    double z2 = b2;
    for (int h = 0; h < H; h++) {
        z2 += W2[h] * a1[h];
    }
    double sortie = sigmoid(z2);
    int prediction = (sortie > 0.5) ? 1 : 0;
    if (prediction == label[i]) correct_final++;
    printf("Point %d : sortie = %.3f, prediction = %d, vraie classe = %d\n", i, sortie, prediction, label[i]);
}
printf("accuracy_test=%.4f\n", (double)correct_final / n);

    return 0;
}