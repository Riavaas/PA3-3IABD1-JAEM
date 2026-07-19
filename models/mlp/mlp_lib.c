/*
 * MLP / PMC — bibliothèque partagée pour ctypes.
 * Les données sont fournies par Python (tableaux plats),
 * aucun fichier n'est lu ici.
 *
 * Architecture : couche cachée sigmoïde (H neurones) + sortie softmax (K=3).
 * Apprentissage : rétropropagation, correction de la couche cachée
 * avec l'ANCIEN W2 (avant sa propre correction).
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define K_CLASSES 3

static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

static void forward(const float *x, int d, int H,
                     const double *W1, const double *b1,
                     const double *W2, const double *b2,
                     double *a1, double *probs) {
    for (int h = 0; h < H; h++) {
        double z = b1[h];
        const double *w = W1 + (size_t)h * d;
        for (int j = 0; j < d; j++) {
            z += w[j] * x[j];
        }
        a1[h] = sigmoid(z);
    }

    double z2[K_CLASSES];
    double maxz = -1e300;
    for (int k = 0; k < K_CLASSES; k++) {
        double z = b2[k];
        const double *w = W2 + (size_t)k * H;
        for (int h = 0; h < H; h++) {
            z += w[h] * a1[h];
        }
        z2[k] = z;
        if (z > maxz) maxz = z;
    }
    double sum = 0.0;
    for (int k = 0; k < K_CLASSES; k++) {
        probs[k] = exp(z2[k] - maxz); 
        sum += probs[k];

    }
    for (int k = 0; k < K_CLASSES; k++) probs[k] /= sum;
}

static int argmaxk(const double *probs) {
    int best = 0;
    for (int k = 1; k < K_CLASSES; k++) if (probs[k] > probs[best]) best = k;
    return best;
}

static void corriger_cachee(const double *dz2, const float *x, int d, int H,
                             const double *a1, double *W1, double *b1,
                             const double *W2, double lr) {
    for (int h = 0; h < H; h++) {
        double da1 = 0.0;
        for (int k = 0; k < K_CLASSES; k++) {
            da1 += dz2[k] * W2[(size_t)k * H + h];
        }
        double dz1 = da1 * a1[h] * (1.0 - a1[h]);
        double *w = W1 + (size_t)h * d;
        for (int j = 0; j < d; j++) {
            w[j] -= lr * dz1 * x[j];
        }
        b1[h] -= lr * dz1;
    }
}

static void corriger_sortie(const double *dz2, const double *a1, int H,
                             double *W2, double *b2, double lr) {
    for (int k = 0; k < K_CLASSES; k++) {
        double *w = W2 + (size_t)k * H;
        for (int h = 0; h < H; h++) {
            w[h] -= lr * dz2[k] * a1[h];
        }
        b2[k] -= lr * dz2[k];
    }
}

void entrainer(const float* X, const int* y, int n, int d, int H, int epochs, double lr, unsigned int seed,
               double* W1_out, double* b1_out, double* W2_out, double* b2_out){

    double *a1 = (double *)malloc((size_t)H * sizeof(double));
    double probs[K_CLASSES];

    srand(seed);
    double scale1 = 1.0 / sqrt((double)d);
    double scale2 = 1.0 / sqrt((double)H);
    for (size_t i = 0; i < (size_t)H * (size_t)d; i++)
        W1_out[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale1;
    for (size_t i = 0; i < (size_t)K_CLASSES * (size_t)H; i++)
        W2_out[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale2;
    for (int h = 0; h < H; h++) b1_out[h] = 0.0;
    for (int k = 0; k < K_CLASSES; k++) b2_out[k] = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < n; i++) {
            const float *x = X + (size_t)i * d;
            forward(x, d, H, W1_out, b1_out, W2_out, b2_out, a1, probs);

            double dz2[K_CLASSES];
            for (int k = 0; k < K_CLASSES; k++) {
                dz2[k] = probs[k] - (k == y[i] ? 1.0 : 0.0);
            }

            corriger_cachee(dz2, x, d, H, a1, W1_out, b1_out, W2_out, lr);   /* avec l'ANCIEN W2 */
            corriger_sortie(dz2, a1, H, W2_out, b2_out, lr);                  /* W2 change maintenant */
        }
    }

    free(a1);
}

void predire_batch(const float* X_test, int n_test, int d, int H,
                   const double* W1, const double* b1,
                   const double* W2, const double* b2,
                   int* predictions_out){

    double *a1 = (double *)malloc((size_t)H * sizeof(double));
    double probs[K_CLASSES];

    for(int i = 0; i < n_test; i++){

        const float *x = X_test + (size_t)i * d;
        forward(x, d, H, W1, b1, W2, b2, a1, probs);
        predictions_out[i] = argmaxk(probs);
    }

    free(a1);

}

