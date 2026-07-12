#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "eigen-5.0.0/Eigen/Dense"

/*
 * RBF Network (version kmeans) pour les features exportées par
 * preprocessing/build_dataset.py (mêmes fichiers que linear_model.c).
 *
 * Fichiers attendus :
 *   datasets/transformed/<variante>/<normalisee|non_normalisee>/
 *     X_train.f32bin, y_train.i32bin, X_test.f32bin, y_test.i32bin
 *
 * Format binaire (little-endian, NE PAS modifier) :
 *   X_*.f32bin : [int32 n][int32 d][float32 n*d]  (row-major)
 *   y_*.i32bin : [int32 n][int32 labels[n]]       (labels dans {0,1,2})
 *
 * Modèle RBF :
 *   1) kmeans -> K centres
 *   2) phi[i][c] = e^(-gamma * ||x_i - centre_c||^2)   (N x K)
 *   3) W = phi^+ * Y   (pseudo inverse, Y en one-hot)
 *   4) pred = argmax_k somme_c W[k][c] * phi_c(x)
 */

#define K_CLASSES 3

extern "C" {

static void die(const char *msg) {
    fprintf(stderr, "Erreur: %s\n", msg);
    exit(1);
}

double** allocation_matricielle(int largeur, int hauteur){
    double** m = (double**)malloc(largeur * sizeof(double*));
    for(int i = 0; i < largeur; i++){
        m[i] = (double*)calloc(hauteur, sizeof(double));
     }
    return m;
}
void liberateur_judiciaire(double** mat, int rows){
    for(int i = 0; i < rows; i++){
        free(mat[i]);
    }
    free(mat);
}
void set_seed(unsigned int s){ srand(s); }

// Inversion Matricielles avec Eigen

double** pseudo_inverse(double** A, int m, int n) {
    Eigen::MatrixXd Mat(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            Mat(i, j) = A[i][j];

    Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(Mat);
    Eigen::MatrixXd Pinv = svd.solve(Eigen::MatrixXd::Identity(m, m));


    double** Aplus = allocation_matricielle(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Aplus[i][j] = Pinv(i, j);

    return Aplus;
}


// distanceQuadratique : e^ -gamma * distance^2
double distanceQuadratique(double* x1, double* x2, int dimension){

    double distance = 0;
    for(int i = 0; i < dimension; i++){
        distance += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return distance;
}

double influence(double distance, double gamma){
   double res = exp(-gamma * distance);
   return res;
}

// phi[i][c] = e^(-gamma * ||x_i - centre_c||^2)  -> matrice N x K
double** remplirMatriceCentres(int dimension, double** matriceEntree, int nombrePoints, double** centres, int nbCentres, double gamma){
    double** matriceSortie = allocation_matricielle(nombrePoints, nbCentres);
    for(int i = 0; i < nombrePoints; i++){
        for(int c = 0; c < nbCentres; c++){
            double distance = distanceQuadratique(matriceEntree[i], centres[c], dimension);
            matriceSortie[i][c] = influence(distance, gamma);
        }
    }
    return matriceSortie;
}

// kmeans : on tire nbCentres points au hasard comme centres de depart
// et on repete chaque point rejoint le centre le plus proche
// et chaque centre devient la moyenne de son groupe
double** kmeans(double** X, int nombrePoints, int dimension, int nbCentres, int maxIterations){
    double** centres = allocation_matricielle(nbCentres, dimension);

    // init : nbCentres points differents pris au hasard
    int* dejaPris = (int*)calloc(nombrePoints, sizeof(int));
    for(int c = 0; c < nbCentres; c++){
        int idx = rand() % nombrePoints;
        while(dejaPris[idx] == 1){
            idx = rand() % nombrePoints;
        }
        dejaPris[idx] = 1;
        for(int j = 0; j < dimension; j++){
            centres[c][j] = X[idx][j];
        }
    }
    free(dejaPris);

    int* groupe = (int*)malloc(nombrePoints * sizeof(int));
    for(int i = 0; i < nombrePoints; i++){ groupe[i] = -1; }

    for(int iteration = 0; iteration < maxIterations; iteration++){
        // assignation : chaque point va au centre le plus proche
        int changement = 0;
        for(int i = 0; i < nombrePoints; i++){
            int meilleur = 0;
            double meilleureDistance = distanceQuadratique(X[i], centres[0], dimension);
            for(int c = 1; c < nbCentres; c++){
                double d = distanceQuadratique(X[i], centres[c], dimension);
                if(d < meilleureDistance){
                    meilleureDistance = d;
                    meilleur = c;
                }
            }
            if(groupe[i] != meilleur){ changement = 1; }
            groupe[i] = meilleur;
        }

        // mise a jour : chaque centre = moyenne des points de son groupe
        for(int c = 0; c < nbCentres; c++){
            double* somme = (double*)calloc(dimension, sizeof(double));
            int taille = 0;
            for(int i = 0; i < nombrePoints; i++){
                if(groupe[i] == c){
                    for(int j = 0; j < dimension; j++){ somme[j] += X[i][j]; }
                    taille++;
                }
            }
            // si le groupe est vide on garde l'ancien centre
            if(taille > 0){
                for(int j = 0; j < dimension; j++){ centres[c][j] = somme[j] / taille; }
            }
            free(somme);
        }

        printf("kmeans iteration %d\n", iteration + 1);
        if(changement == 0){ break; }  // plus rien ne bouge = convergence
    }
    free(groupe);
    return centres;
}

// W[k][i] = poids du centre i pour la classe k
double** calcul_poids(double** matInv, double** y_true, int nb_centre, int nb_point, int nb_classes){
    double** w = allocation_matricielle(nb_classes, nb_centre);
    for (int classe = 0; classe < nb_classes; classe++){
        for (int centre = 0; centre < nb_centre; centre++){
            for (int exemple = 0; exemple < nb_point; exemple++){
                w[classe][centre] += matInv[centre][exemple] * y_true[exemple][classe];
            }
        }
    }
    return w;
}

// prediction : score_k = somme_c w[k][c] * e^(-gamma * ||x - centre_c||^2)
// on renvoie la classe avec le plus grand score
int predict(double* x, double** centres, int nbCentres, double** w, int nbClasses, int dimension, double gamma){
    int meilleureClasse = 0;
    double meilleurScore = 0;
    for(int k = 0; k < nbClasses; k++){
        double score = 0;
        for(int c = 0; c < nbCentres; c++){
            double phi = influence(distanceQuadratique(x, centres[c], dimension), gamma);
            score += w[k][c] * phi;
        }
        if(k == 0 || score > meilleurScore){
            meilleurScore = score;
            meilleureClasse = k;
        }
    }
    return meilleureClasse;
}

// chargement des fichiers binaires (repris de linear_model.c)
// on stocke direct en double pour reutiliser les fonctions du dessus
static void load_X(const char *path, int *out_n, int *out_d, double **out_X) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        die("Impossible d'ouvrir X");
    }

    int32_t n32 = 0, d32 = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1) die("Lecture n échouée (X)");
    if (fread(&d32, sizeof(int32_t), 1, f) != 1) die("Lecture d échouée (X)");
    if (n32 <= 0 || d32 <= 0) die("n/d invalides (X)");

    size_t n = (size_t)n32;
    size_t d = (size_t)d32;
    double *X = (double *)malloc(n * d * sizeof(double));
    if (!X) die("malloc X");

    // le fichier est en float32, on convertit ligne par ligne (repris de linear_model.c)
    float *ligne = (float *)malloc(d * sizeof(float));
    if (!ligne) die("malloc ligne");
    for (size_t i = 0; i < n; i++) {
        if (fread(ligne, sizeof(float), d, f) != d) die("Lecture data échouée (X)");
        for (size_t j = 0; j < d; j++) {
            X[i * d + j] = (double)ligne[j];
        }
    }
    free(ligne);
    fclose(f);

    *out_n = (int)n32;
    *out_d = (int)d32;
    *out_X = X;
}

static void load_y(const char *path, int expected_n, int **out_y) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        die("Impossible d'ouvrir y");
    }

    int32_t n32 = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1) die("Lecture n échouée (y)");
    if (n32 != expected_n) die("n incohérent entre X et y");
    if (n32 <= 0) die("n invalide (y)");

    size_t n = (size_t)n32;
    int *y = (int *)malloc(n * sizeof(int));
    if (!y) die("malloc y");

    int32_t tmp = 0;
    for (size_t i = 0; i < n; i++) {
        if (fread(&tmp, sizeof(int32_t), 1, f) != 1) die("Lecture label échouée (y)");
        if (tmp < 0 || tmp >= K_CLASSES) die("Label hors plage (attendu 0..2)");
        y[i] = (int)tmp;
    }
    fclose(f);
    *out_y = y;
}

// accuracy sur un jeu X/y quelconque (train ou test)
static float accuracy(double** lignes, const int *y, int n, double** centres, int nbCentres, double** w, int d, double gamma) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        int pred = predict(lignes[i], centres, nbCentres, w, K_CLASSES, d, gamma);
        if (pred == y[i]) correct++;
    }
    return (float)correct / (float)n;
}

static void confusion_test(double** lignes, const int *y, int n, double** centres, int nbCentres, double** w, int d, double gamma, int conf[K_CLASSES][K_CLASSES]) {
    for (int a = 0; a < K_CLASSES; a++) {
        for (int p = 0; p < K_CLASSES; p++) {
            conf[a][p] = 0;
        }
    }
    for (int i = 0; i < n; i++) {
        int pred = predict(lignes[i], centres, nbCentres, w, K_CLASSES, d, gamma);
        int yi = y[i];
        if (yi >= 0 && yi < K_CLASSES && pred >= 0 && pred < K_CLASSES) {
            conf[yi][pred] += 1;
        }
    }
}

int main(int argc, char *argv[]) {
    // Par défaut : variante NB normalisée (d=4096).
    const char *x_train_path = "datasets/transformed/nb/normalisee/X_train.f32bin";
    const char *y_train_path = "datasets/transformed/nb/normalisee/y_train.i32bin";
    const char *x_test_path  = "datasets/transformed/nb/normalisee/X_test.f32bin";
    const char *y_test_path  = "datasets/transformed/nb/normalisee/y_test.i32bin";
    double gamma = 0.01;
    int nbCentres = 100;
    unsigned int seed = 42;

    if (argc >= 5) {
        x_train_path = argv[1];
        y_train_path = argv[2];
        x_test_path  = argv[3];
        y_test_path  = argv[4];
    }
    if (argc >= 6) gamma = atof(argv[5]);
    if (argc >= 7) nbCentres = atoi(argv[6]);
    if (argc >= 8) seed = (unsigned int)atoi(argv[7]);

    int n_train = 0, d_train = 0;
    int n_test = 0, d_test = 0;
    double *X_train = NULL;
    double *X_test = NULL;
    int *y_train = NULL;
    int *y_test = NULL;

    load_X(x_train_path, &n_train, &d_train, &X_train);
    load_y(y_train_path, n_train, &y_train);
    load_X(x_test_path, &n_test, &d_test, &X_test);
    load_y(y_test_path, n_test, &y_test);

    if (d_train != d_test) {
        die("Les jeux train et test doivent avoir le même nombre de features");
    }
    int d = d_train;
    if (nbCentres > n_train) die("nb_centres > n_train, pas possible");

    printf("Chargé train: n=%d, d=%d\n", n_train, d);
    printf("Chargé test : n=%d, d=%d\n", n_test, d);
    printf("Entraînement RBF kmeans, gamma=%g, nb_centres=%d, seed=%u\n", gamma, nbCentres, seed);

    // tableaux de pointeurs vers chaque ligne (pour reutiliser les fonctions en double**)
    double** lignes_train = (double**)malloc(n_train * sizeof(double*));
    for (int i = 0; i < n_train; i++) { lignes_train[i] = X_train + (size_t)i * (size_t)d; }
    double** lignes_test = (double**)malloc(n_test * sizeof(double*));
    for (int i = 0; i < n_test; i++) { lignes_test[i] = X_test + (size_t)i * (size_t)d; }

    // Y en one-hot (n_train x K)
    double** Y = allocation_matricielle(n_train, K_CLASSES);
    for (int i = 0; i < n_train; i++) {
        Y[i][y_train[i]] = 1.0;
    }

    set_seed(seed);

    // 1) kmeans
    double** centres = kmeans(lignes_train, n_train, d, nbCentres, 50);

    // 2) phi (N x K)  3) pseudo inverse (K x N)  4) W = phi^+ * Y
    double** phi = remplirMatriceCentres(d, lignes_train, n_train, centres, nbCentres, gamma);
    double** phiInverse = pseudo_inverse(phi, n_train, nbCentres);
    double** w = calcul_poids(phiInverse, Y, nbCentres, n_train, K_CLASSES);

    float acc_train = accuracy(lignes_train, y_train, n_train, centres, nbCentres, w, d, gamma);
    float acc_test = accuracy(lignes_test, y_test, n_test, centres, nbCentres, w, d, gamma);
    printf("acc train %.3f\n", acc_train);
    printf("acc test %.3f\n", acc_test);

    int conf[K_CLASSES][K_CLASSES];
    confusion_test(lignes_test, y_test, n_test, centres, nbCentres, w, d, gamma, conf);
    printf("confusion\n");
    for (int a = 0; a < K_CLASSES; a++) {
        printf("%d %d %d\n", conf[a][0], conf[a][1], conf[a][2]);
    }

    // menage
    liberateur_judiciaire(Y, n_train);
    liberateur_judiciaire(centres, nbCentres);
    liberateur_judiciaire(phi, n_train);
    liberateur_judiciaire(phiInverse, nbCentres);
    liberateur_judiciaire(w, K_CLASSES);
    free(lignes_train);
    free(lignes_test);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    return 0;
}}
