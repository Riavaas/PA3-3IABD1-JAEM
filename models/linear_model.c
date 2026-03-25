#include <stdio.h>

#define MAX_POINTS 100
//https://numiqo.fr/tutorial/linear-regression


/*
 * Régression linéaire simple : on cherche une droite Y ≈ b*X + a
 * qui colle au mieux aux points (moindres carrés).
 *
 * X = variable indépendante (ex. taille en mètres)
 * Y = variable dépendante (ex. poids en kg)
 * b = pente de la droite, a = valeur de Y quand X = 0 (ordonnée à l’origine)
 *
 * En vrai : Y = b*X + a + epsilon  ;  epsilon = résidu (erreur)
 */

typedef struct {
    double x;
    double y;
} DataPoint;

int main(int argc, char *argv[]) {
    DataPoint points[MAX_POINTS];
    int n = 0;

    /* Sans argument : saisie au clavier. Avec un argument : lecture dans un fichier texte
       (pour les données venant d’Excel : exporter/mettre une ligne = X puis Y, cf. test_points.txt) */
    FILE *in = stdin;
    if (argc >= 2) {
        in = fopen(argv[1], "r");
        if (!in) {
            perror(argv[1]);
            return 1;
        }
    } else {
        printf("Entrez X et Y (deux nombres par ligne). Fin : -1 -1\n");
    }

    while (n < MAX_POINTS) {
        double x, y;
        if (fscanf(in, "%lf %lf", &x, &y) != 2) {
            break;
        }
        if (in == stdin && x == -1 && y == -1) {
            break;
        }
        points[n].x = x;
        points[n].y = y;
        n++;
    }

    if (in != stdin) {
        fclose(in);
    }

    if (n < 2) {
        printf("Il faut au moins 2 points.\n");
        return 1;
    }

    printf("\nPoints :\n");
    for (int i = 0; i < n; i++) {
        printf("Point %d : (%.2f, %.2f)\n", i + 1, points[i].x, points[i].y);
    }

    /* Formules des moindres carrés (n points) */
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += points[i].x;
        sy += points[i].y;
        sxx += points[i].x * points[i].x;
        sxy += points[i].x * points[i].y;
    }
    double dn = (double)n;
    double b = (dn * sxy - sx * sy) / (dn * sxx - sx * sx);
    double a = (sy - b * sx) / dn;

    printf("\nDroite : Y = %.4f * X + %.4f\n", b, a);
    return 0;
}
