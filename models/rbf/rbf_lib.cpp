#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "eigen-5.0.0/Eigen/Dense"

/*
 * RBF Network (version kmeans) — bibliothèque partagée pour ctypes.
 * Les données sont fournies par Python (tableaux plats en double),
 * aucun fichier n'est lu ici.
 *
 * Modèle RBF :
 *   1) kmeans -> K centres
 *   2) phi[i][c] = e^(-gamma * ||x_i - centre_c||^2)   (N x K)
 *   3) W = phi^+ * Y   (pseudo inverse, Y en one-hot)
 *   4) pred = argmax_k somme_c W[k][c] * phi_c(x)
 */

#define K_CLASSES 3

extern "C" {

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

void entrainer(const double* X, const int* y, int n, int d, int nb_centres, double gamma, unsigned int seed, double* centres_out, double* W_out ){

    set_seed(seed);

    double** lignes = (double**)malloc(n * sizeof(double*));

    for (int i = 0; i < n; i++) { 
        lignes[i] = (double*)X + (size_t)i * (size_t)d; 
    }

    double** Y = allocation_matricielle(n, K_CLASSES);

    for (int i = 0; i < n; i++){
        Y[i][y[i]] = 1.0;
    }

    double** centres = kmeans(lignes, n, d, nb_centres, 50);
    double** phi = remplirMatriceCentres(d, lignes, n, centres, nb_centres, gamma);
    double** phiInverse = pseudo_inverse(phi, n, nb_centres);
    double** w = calcul_poids(phiInverse, Y, nb_centres, n, K_CLASSES);

    for(int c = 0; c < nb_centres; c++){
            for (int j = 0; j < d; j++){
                centres_out[c * d + j] = centres[c][j];
        }
    }

    for(int c = 0; c < K_CLASSES; c++){
        for (int j = 0; j < nb_centres; j++){
            W_out[c * nb_centres + j] = w[c][j];
        }
    }

    liberateur_judiciaire(Y, n);
    liberateur_judiciaire(centres, nb_centres);
    liberateur_judiciaire(phi, n);
    liberateur_judiciaire(phiInverse, nb_centres);
    liberateur_judiciaire(w, K_CLASSES);
    free(lignes);

}

void predire_batch(const double* X_test, int n_test, int d, const double* centres_plat, int nb_centres, const double* W_plat, double gamma, int* predictions_out)
{
    double** lignes = (double**)malloc(n_test * sizeof(double*));
    for (int i = 0; i < n_test; i++) {
        lignes[i] = (double*)X_test + (size_t)i * (size_t)d;
    }

    double** centres = (double**)malloc(nb_centres * sizeof(double*));
    for (int c = 0; c < nb_centres; c++) {
        centres[c] = (double*)centres_plat + (size_t)c * (size_t)d;
    }

    double** w = (double**)malloc(K_CLASSES * sizeof(double*));
    for (int k = 0; k < K_CLASSES; k++) {
        w[k] = (double*)W_plat + (size_t)k * (size_t)nb_centres;
    }

    for (int i = 0; i < n_test; i++) {
        predictions_out[i] = predict(lignes[i], centres, nb_centres, w, K_CLASSES, d, gamma);
    }

    free(lignes);
    free(centres);
    free(w);
}

}
