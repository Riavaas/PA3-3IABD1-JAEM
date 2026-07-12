#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "eigen-5.0.0/Eigen/Dense"

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

// Header magique (flemme de faire un header séparé)
double** allocation_matricielle(int largeur, int hauteur);
void liberateur_judiciaire(double** mat, int rows);

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



//double
   /* e^ -gamma * distance^2*/
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
// version naive : les centres = tous les points, donc matrice N x N
double** remplirMatrice(int dimension, double** matriceEntree, int nombrePoints, double gamma){
    double** matriceSortie = (double**)malloc(nombrePoints * sizeof(double*));
    for(int i = 0; i < nombrePoints; i++){
        matriceSortie[i] = (double*)malloc(nombrePoints * sizeof(double));
    }
    for(int i = 0; i < nombrePoints; i++){
        for(int j = 0; j < nombrePoints; j++){
            double distance = distanceQuadratique(matriceEntree[i], matriceEntree[j], dimension);
            matriceSortie[i][j] = influence(distance, gamma);
        }
    }


    return matriceSortie;
}
// pareil que remplirMatrice mais avec des centres a part (version kmeans)
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

        if(changement == 0){ break; }  // plus rien ne bouge -> convergence
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
//  renvoie  classe avec le plus grand score
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



int main(){
    double gamma = 0.008;
    int nombrePoints = 6;
    int dimension = 3;
    int nbclasses = 3;

    //matrices à la mano
    double p0[] = {5.0, 2.0, 9.0};
    double p1[] = {2.0, 6.0, 3.0};
    double p2[] = {4.0, 4.0, 5.0};
    double p3[] = {6.0, 1.0, 2.0};
    double p4[] = {3.0, 8.0, 7.0};
    double p5[] = {9.0, 2.0, 1.0};
    double* X[] = {p0, p1, p2, p3, p4, p5};

    double y0[] = {1, 0, 0};
    double y1[] = {1, 0, 0};
    double y2[] = {0, 1, 0};
    double y3[] = {0, 1, 0};
    double y4[] = {0, 0, 1};
    double y5[] = {0, 0, 1};
    double* Y[] = {y0, y1, y2, y3, y4, y5};

    set_seed(42);

    // version naive : centres = tous les points
    // 1) phi (N x N)   2) pseudo inverse   3) W = phi^+ * Y
    double** influenceMat = remplirMatrice(dimension, X, nombrePoints, gamma);
    double** influenceInverse = pseudo_inverse(influenceMat, nombrePoints, nombrePoints);
    double** w = calcul_poids(influenceInverse, Y, nombrePoints, nombrePoints, nbclasses);

    printf("matrice phi (naive) :\n");
    for(int i = 0; i < nombrePoints; i++){
        for(int j = 0; j < nombrePoints; j++){
            printf("%f ", influenceMat[i][j]);
        }
        printf("\n");
    }

    printf("\n--- version naive (centres = les %d points) ---\n", nombrePoints);
    int bon = 0;
    for(int i = 0; i < nombrePoints; i++){
        // ici, centres =  points eux memes
        int pred = predict(X[i], X, nombrePoints, w, nbclasses, dimension, gamma);
        int vraie = 0;
        for(int k = 0; k < nbclasses; k++){
            if(Y[i][k] == 1){ vraie = k; }
        }
        printf("point %d : vraie classe %d, predit %d\n", i, vraie, pred);
        if(pred == vraie){ bon++; }
    }
    printf("accuracy naive : %f\n", (double)bon / nombrePoints);

    // version kmeans : K centres < N points
    int nbCentres = 4;
    double** centres = kmeans(X, nombrePoints, dimension, nbCentres, 100);
    double** phi = remplirMatriceCentres(dimension, X, nombrePoints, centres, nbCentres, gamma);
    // phi est N x K donc la pseudo inverse est K x N
    double** phiInverse = pseudo_inverse(phi, nombrePoints, nbCentres);
    double** wKmeans = calcul_poids(phiInverse, Y, nbCentres, nombrePoints, nbclasses);

    printf("\n--- version kmeans (%d centres) ---\n", nbCentres);
    bon = 0;
    for(int i = 0; i < nombrePoints; i++){
        int pred = predict(X[i], centres, nbCentres, wKmeans, nbclasses, dimension, gamma);
        int vraie = 0;
        for(int k = 0; k < nbclasses; k++){
            if(Y[i][k] == 1){ vraie = k; }
        }
        printf("point %d : vraie classe %d, predit %d\n", i, vraie, pred);
        if(pred == vraie){ bon++; }
    }
    printf("accuracy kmeans : %f\n", (double)bon / nombrePoints);

    // menage
    liberateur_judiciaire(influenceMat, nombrePoints);
    liberateur_judiciaire(influenceInverse, nombrePoints);
    liberateur_judiciaire(w, nbclasses);
    liberateur_judiciaire(centres, nbCentres);
    liberateur_judiciaire(phi, nombrePoints);
    liberateur_judiciaire(phiInverse, nbCentres);
    liberateur_judiciaire(wKmeans, nbclasses);

    return 0;
}}
