/*
 * Perceptron multi-couches (PMC / MLP) pour classification multi-classes (K=3),
 * pour les features exportées par preprocessing/build_dataset.py (même format
 * que linear_model.c).
 *
 * Vient corriger l'échec du modèle linéaire (linear_model.c) sur les données
 * non linéairement séparables : au lieu d'une seule étape (pixels -> score),
 * on ajoute une étape intermédiaire qui transforme les pixels avant la décision
 * finale.
 *
 * Architecture en deux étapes :
 *   1) Couche cachée : chaque neurone reçoit TOUTE l'image (tous les pixels),
 *      avec un poids personnel par pixel + un biais. Il multiplie chaque pixel
 *      par son poids, additionne tout, ajoute le biais -> un seul nombre par
 *      neurone.
 *   2) Couche de sortie : reçoit UNIQUEMENT les nombres produits par la couche
 *      cachée (jamais les pixels bruts). Fait le même genre de calcul (poids +
 *      biais, multiplié-additionné) pour obtenir un score par classe.
 *      La classe prédite est celle avec le score le plus haut.
 *
 * Meme mecanique que mlp_csv.c (deja teste, 100% sur linear.csv et xor.csv),
 * generalisee de 1 sortie (sigmoide) a K=3 sorties (softmax), et de tableaux
 * fixes a des fichiers binaires reels charges dynamiquement.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define K_CLASSES 3

static void die(const char *msg) {
    fprintf(stderr, "Erreur: %s\n", msg);
    exit(1);
}

static size_t checked_mul_size(size_t a, size_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > (SIZE_MAX / b)) die("Overflow allocation");
    return a * b;
}

/* Format binaire (identique a linear_model.c) :
 *   X_*.f32bin : [int32 n][int32 d][float32 n*d]
 *   y_*.i32bin : [int32 n][int32 labels[n]] dans {0,1,2} */
static void load_X(const char *path, int *out_n, int *out_d, float **out_X) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); die("Impossible d'ouvrir X"); }
    int32_t n32 = 0, d32 = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1) die("Lecture n echouee (X)");
    if (fread(&d32, sizeof(int32_t), 1, f) != 1) die("Lecture d echouee (X)");
    if (n32 <= 0 || d32 <= 0) die("n/d invalides (X)");
    size_t n = (size_t)n32, d = (size_t)d32;
    size_t count = checked_mul_size(n, d);
    float *X = (float *)malloc(count * sizeof(float));
    if (!X) die("malloc X");
    if (fread(X, sizeof(float), count, f) != count) die("Lecture data echouee (X)");
    fclose(f);
    *out_n = (int)n32; *out_d = (int)d32; *out_X = X;
}

static void load_y(const char *path, int expected_n, int **out_y) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); die("Impossible d'ouvrir y"); }
    int32_t n32 = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1) die("Lecture n echouee (y)");
    if (n32 != expected_n) die("n incoherent entre X et y");
    size_t n = (size_t)n32;
    int *y = (int *)malloc(n * sizeof(int));
    if (!y) die("malloc y");
    int32_t tmp = 0;
    for (size_t i = 0; i < n; i++) {
        if (fread(&tmp, sizeof(int32_t), 1, f) != 1) die("Lecture label echouee (y)");
        if (tmp < 0 || tmp >= K_CLASSES) die("Label hors plage (attendu 0..2)");
        y[i] = (int)tmp;
    }
    fclose(f);
    *out_y = y;
}

static double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

/* Forward pass pour un exemple : remplit a1 (H valeurs) et probs (K_CLASSES). */
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
        probs[k] = exp(z2[k] - maxz);   /* -maxz : stabilite numerique du softmax */
        sum += probs[k];
    }
    for (int k = 0; k < K_CLASSES; k++) probs[k] /= sum;
}

static int argmaxk(const double *probs) {
    int best = 0;
    for (int k = 1; k < K_CLASSES; k++) if (probs[k] > probs[best]) best = k;
    return best;
}

/* Corrige W1/b1 en utilisant l'ANCIEN W2 (avant sa propre correction) —
 * meme piege d'ordre que dans mlp_csv.c, voir JOURNAL.md. */
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

int main(int argc, char *argv[]) {
    const char *x_train_path = "datasets/transformed/nb/normalisee/X_train.f32bin";
    const char *y_train_path = "datasets/transformed/nb/normalisee/y_train.i32bin";
    const char *x_test_path  = "datasets/transformed/nb/normalisee/X_test.f32bin";
    const char *y_test_path  = "datasets/transformed/nb/normalisee/y_test.i32bin";
    int H = 32;
    int epochs = 30;
    double lr = 0.001;
    unsigned int seed = 67;

    if (argc >= 5) {
        x_train_path = argv[1]; y_train_path = argv[2];
        x_test_path  = argv[3]; y_test_path  = argv[4];
    }
    if (argc >= 6) H = atoi(argv[5]);
    if (argc >= 7) epochs = atoi(argv[6]);
    if (argc >= 8) lr = atof(argv[7]);
    if (argc >= 9) seed = (unsigned int)atoi(argv[8]);

    int n_train = 0, d_train = 0, n_test = 0, d_test = 0;
    float *X_train = NULL, *X_test = NULL;
    int *y_train = NULL, *y_test = NULL;

    load_X(x_train_path, &n_train, &d_train, &X_train);
    load_y(y_train_path, n_train, &y_train);
    load_X(x_test_path, &n_test, &d_test, &X_test);
    load_y(y_test_path, n_test, &y_test);
    if (d_train != d_test) die("d incoherent entre train et test");
    int d = d_train;

    printf("Charge train: n=%d, d=%d\n", n_train, d);
    printf("Charge test : n=%d, d=%d\n", n_test, d);
    printf("PMC : H=%d, epochs=%d, lr=%g\n", H, epochs, lr);

    double *W1 = (double *)malloc(checked_mul_size((size_t)H, (size_t)d) * sizeof(double));
    double *b1 = (double *)calloc((size_t)H, sizeof(double));
    double *W2 = (double *)malloc(checked_mul_size((size_t)K_CLASSES, (size_t)H) * sizeof(double));
    double *b2 = (double *)calloc((size_t)K_CLASSES, sizeof(double));
    if (!W1 || !b1 || !W2 || !b2) die("malloc modele");

    srand(seed);
    double scale1 = 1.0 / sqrt((double)d);
    double scale2 = 1.0 / sqrt((double)H);
    for (size_t i = 0; i < (size_t)H * (size_t)d; i++)
        W1[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale1;
    for (size_t i = 0; i < (size_t)K_CLASSES * (size_t)H; i++)
        W2[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale2;

    double *a1 = (double *)malloc((size_t)H * sizeof(double));
    double probs[K_CLASSES];

    for (int epoch = 0; epoch < epochs; epoch++) {
        int correct_train = 0;
        for (int i = 0; i < n_train; i++) {
            const float *x = X_train + (size_t)i * d;
            forward(x, d, H, W1, b1, W2, b2, a1, probs);
            int pred = argmaxk(probs);
            if (pred == y_train[i]) correct_train++;

            double dz2[K_CLASSES];
            for (int k = 0; k < K_CLASSES; k++) {
                dz2[k] = probs[k] - (k == y_train[i] ? 1.0 : 0.0);
            }

            corriger_cachee(dz2, x, d, H, a1, W1, b1, W2, lr);   /* avec l'ANCIEN W2 */
            corriger_sortie(dz2, a1, H, W2, b2, lr);              /* W2 change maintenant */
        }

        int correct_test = 0;
        for (int i = 0; i < n_test; i++) {
            const float *x = X_test + (size_t)i * d;
            forward(x, d, H, W1, b1, W2, b2, a1, probs);
            if (argmaxk(probs) == y_test[i]) correct_test++;
        }

        printf("epoch %d train %.3f test %.3f\n", epoch + 1,
               (double)correct_train / n_train, (double)correct_test / n_test);
    }

    int conf[K_CLASSES][K_CLASSES] = {{0}};
    for (int i = 0; i < n_test; i++) {
        const float *x = X_test + (size_t)i * d;
        forward(x, d, H, W1, b1, W2, b2, a1, probs);
        conf[y_test[i]][argmaxk(probs)]++;
    }
    printf("confusion\n");
    for (int a = 0; a < K_CLASSES; a++) {
        printf("%d %d %d\n", conf[a][0], conf[a][1], conf[a][2]);
    }

    /* Sauvegarde des poids dans un fichier texte simple, lisible.
     * Ligne 1 : H d K_CLASSES (les dimensions, necessaires pour recharger).
     * Puis W1 (H*d valeurs), b1 (H), W2 (K*H), b2 (K), chacun sur sa ligne. */
    const char *poids_path = "models/mlp/poids_mlp.txt";
    FILE *fp = fopen(poids_path, "w");
    if (!fp) {
        perror(poids_path);
    } else {
        fprintf(fp, "%d %d %d\n", H, d, K_CLASSES);
        for (size_t i = 0; i < (size_t)H * (size_t)d; i++) fprintf(fp, "%.6f ", W1[i]);
        fprintf(fp, "\n");
        for (int h = 0; h < H; h++) fprintf(fp, "%.6f ", b1[h]);
        fprintf(fp, "\n");
        for (size_t i = 0; i < (size_t)K_CLASSES * (size_t)H; i++) fprintf(fp, "%.6f ", W2[i]);
        fprintf(fp, "\n");
        for (int k = 0; k < K_CLASSES; k++) fprintf(fp, "%.6f ", b2[k]);
        fprintf(fp, "\n");
        fclose(fp);
        printf("Poids sauvegardes dans %s\n", poids_path);
    }

    free(W1); free(b1); free(W2); free(b2); free(a1);
    free(X_train); free(X_test); free(y_train); free(y_test);
    return 0;
}
