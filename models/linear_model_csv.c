#include <stdio.h>
#include <stdlib.h>

/*
 * Modele lineaire (perceptron) a 2 classes, qui lit un fichier CSV.
 *
 * Le CSV doit avoir cette forme (1 ligne d'en-tete + 1 ligne par point) :
 *   x1,x2,label
 *   1,1,0
 *   6,6,1
 *
 * Idee : on cherche une droite   w1*x1 + w2*x2 + b = 0.
 *   - si  w1*x1 + w2*x2 + b > 0  -> on predit la classe 1
 *   - sinon                      -> on predit la classe 0
 * On ajuste w1, w2, b a chaque fois qu'on se trompe (regle du perceptron).
 *
 * A la fin, on ECRIT les poids trouves dans "models/poids.txt"
 * pour que le script Python puisse tracer la droite.
 */

#define MAX_POINTS 100

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s fichier.csv [epochs] [lr]\n", argv[0]);
        return 1;
    }
    const char *chemin = argv[1];
    int epochs = (argc >= 3) ? atoi(argv[2]) : 20;     // nombre de passages sur les donnees
    double lr = (argc >= 4) ? atof(argv[3]) : 0.1;     // vitesse d'apprentissage

    /* --- 1) Lecture du fichier CSV --- */
    FILE *f = fopen(chemin, "r");
    if (!f) {
        perror(chemin);
        return 1;
    }

    double x1[MAX_POINTS], x2[MAX_POINTS];   // coordonnees des points
    int label[MAX_POINTS];                   // classe de chaque point (0 ou 1)
    int n = 0;                               // nombre de points lus

    char entete[100];
    fgets(entete, sizeof(entete), f);        // on lit et on jette la ligne d'en-tete

    // on lit chaque ligne "x1,x2,label" tant qu'on y arrive
    while (n < MAX_POINTS && fscanf(f, "%lf,%lf,%d", &x1[n], &x2[n], &label[n]) == 3) {
        n++;
    }
    fclose(f);
    printf("Charge : %d points depuis %s\n", n, chemin);

    /* --- 2) Entrainement --- */
    double w1 = 0.0, w2 = 0.0, b = 0.0;      // parametres de la droite, au depart a 0

    for (int e = 0; e < epochs; e++) {
        int erreurs = 0;
        for (int i = 0; i < n; i++) {
            double y = w1 * x1[i] + w2 * x2[i] + b;
            int pred = (y > 0) ? 1 : 0;          // prediction du modele
            if (pred != label[i]) {              // s'il se trompe, on corrige
                int err = label[i] - pred;       // vaut +1 ou -1
                w1 += lr * err * x1[i];
                w2 += lr * err * x2[i];
                b  += lr * err;
                erreurs++;
            }
        }

        // accuracy = proportion de points bien classes apres cette epoch
        int correct = 0;
        for (int i = 0; i < n; i++) {
            double y = w1 * x1[i] + w2 * x2[i] + b;
            int pred = (y > 0) ? 1 : 0;
            if (pred == label[i]) correct++;
        }
        double acc = (double)correct / (double)n;
        printf("Epoch %d/%d : erreurs=%d, accuracy=%.2f\n", e + 1, epochs, erreurs, acc);

        if (erreurs == 0) break;   // plus aucune erreur : c'est gagne, on s'arrete
    }

    printf("Droite trouvee : %.3f*x1 + %.3f*x2 + %.3f = 0\n", w1, w2, b);

    /* --- 3) On ecrit les poids dans un fichier (pour le graphe Python) --- */
    FILE *fp = fopen("models/poids.txt", "w");
    if (!fp) {
        perror("models/poids.txt");
        return 1;
    }
    fprintf(fp, "%f %f %f\n", w1, w2, b);   // une ligne : w1 w2 b
    fclose(fp);
    printf("Poids ecrits dans models/poids.txt\n");

    return 0;
}