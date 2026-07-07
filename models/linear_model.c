#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Classifieur linéaire multi-classes (K=3) pour les features exportées par
 * preprocessing/build_dataset.py (split train/test déjà fait en Python).
 *
 * Fichiers attendus (par variante et normalisation) :
 *   datasets/transformed/<variante>/<normalisee|non_normalisee>/
 *     X_train.f32bin, y_train.i32bin, X_test.f32bin, y_test.i32bin
 *
 * Format binaire (little-endian, NE PAS modifier) :
 *   X_*.f32bin : [int32 n][int32 d][float32 n*d]  (row-major)
 *   y_*.i32bin : [int32 n][int32 labels[n]]       (labels dans {0,1,2})
 *
 * Modèle (perceptron multi-classes) :
 *   score_k = w_k · x + b_k
 *   pred = argmax_k score_k
 *   Si pred != y: w_y += lr*x ; w_pred -= lr*x ; b_y += lr ; b_pred -= lr
 */

#define K_CLASSES 3

static void die(const char *msg) {
    fprintf(stderr, "Erreur: %s\n", msg);
    exit(1);
}

static size_t checked_mul_size(size_t a, size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > (SIZE_MAX / b)) {
        die("Overflow allocation");
    }
    return a * b;
}

static float dot(const float *w, const float *x, int d) {
    // Produit scalaire w·x (coeur du modèle linéaire).
    float s = 0.0f;
    for (int j = 0; j < d; j++) {
        s += w[j] * x[j];
    }
    return s;
}

static int argmax3(const float s[K_CLASSES]) {
    // Renvoie l'indice du score maximum (classe prédite).
    int best = 0;
    for (int k = 1; k < K_CLASSES; k++) {
        if (s[k] > s[best]) {
            best = k;
        }
    }
    return best;
}

static void load_X(const char *path, int *out_n, int *out_d, float **out_X) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        die("Impossible d'ouvrir X");
    }

    int32_t n32 = 0, d32 = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1) die("Lecture n échouée (X)");
    if (fread(&d32, sizeof(int32_t), 1, f) != 1) die("Lecture d échouée (X)");
    if (n32 <= 0 || d32 <= 0) die("n/d invalides (X)");

    // Allocation du bloc X (n*d float32). Layout = row-major.
    size_t n = (size_t)n32;
    size_t d = (size_t)d32;
    size_t count = checked_mul_size(n, d);
    float *X = (float *)malloc(count * sizeof(float));
    if (!X) die("malloc X");

    if (fread(X, sizeof(float), count, f) != count) die("Lecture data échouée (X)");
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

    // Allocation du vecteur y (n labels) en mémoire.
    size_t n = (size_t)n32;
    int *y = (int *)malloc(n * sizeof(int));
    if (!y) die("malloc y");

    // y a été écrit en int32 dans le script Python
    int32_t tmp = 0;
    for (size_t i = 0; i < n; i++) {
        if (fread(&tmp, sizeof(int32_t), 1, f) != 1) die("Lecture label échouée (y)");
        if (tmp < 0 || tmp >= K_CLASSES) die("Label hors plage (attendu 0..2)");
        y[i] = (int)tmp;
    }
    fclose(f);
    *out_y = y;
}

static float accuracy(const float *W, const float *b, const float *X, const int *y, int n, int d) {
    // Accuracy sur un jeu X/y quelconque (train ou test).
    int correct = 0;
    for (int i = 0; i < n; i++) {
        const float *xi = X + (size_t)i * (size_t)d;
        float s[K_CLASSES];
        for (int k = 0; k < K_CLASSES; k++) {
            s[k] = dot(W + (size_t)k * (size_t)d, xi, d) + b[k];
        }
        int pred = argmax3(s);
        if (pred == y[i]) correct++;
    }
    return (float)correct / (float)n;
}

int main(int argc, char *argv[]) {
    // Par défaut : variante NB normalisée (d=4096).
    const char *x_train_path = "datasets/transformed/nb/normalisee/X_train.f32bin";
    const char *y_train_path = "datasets/transformed/nb/normalisee/y_train.i32bin";
    const char *x_test_path  = "datasets/transformed/nb/normalisee/X_test.f32bin";
    const char *y_test_path  = "datasets/transformed/nb/normalisee/y_test.i32bin";
    int epochs = 30;
    float lr = 0.1f;

    if (argc >= 5) {
        x_train_path = argv[1];
        y_train_path = argv[2];
        x_test_path  = argv[3];
        y_test_path  = argv[4];
    }
    if (argc >= 6) epochs = atoi(argv[5]);
    if (argc >= 7) lr = (float)atof(argv[6]);

    int n_train = 0, d_train = 0;
    int n_test = 0, d_test = 0;
    float *X_train = NULL;
    float *X_test = NULL;
    int *y_train = NULL;
    int *y_test = NULL;

    load_X(x_train_path, &n_train, &d_train, &X_train);
    load_y(y_train_path, n_train, &y_train);
    load_X(x_test_path, &n_test, &d_test, &X_test);
    load_y(y_test_path, n_test, &y_test);

    if (d_train != d_test) {
        fprintf(stderr, "Erreur: d incohérent (train=%d, test=%d)\n", d_train, d_test);
        die("Les jeux train et test doivent avoir le même nombre de features");
    }

    int d = d_train;
    printf("Chargé train: n=%d, d=%d\n", n_train, d);
    printf("Chargé test : n=%d, d=%d\n", n_test, d);
    printf("Entraînement perceptron multi-classes (K=3), epochs=%d, lr=%.4f\n", epochs, lr);

    // Paramètres du modèle: W (K*d) et b (K). Init à 0.
    size_t wd = checked_mul_size((size_t)K_CLASSES, (size_t)d);
    float *W = (float *)calloc(wd, sizeof(float));
    float b[K_CLASSES] = {0.0f, 0.0f, 0.0f};
    if (!W) die("calloc W");

    for (int e = 0; e < epochs; e++) {
        int updates = 0;
        for (int i = 0; i < n_train; i++) {
            const float *xi = X_train + (size_t)i * (size_t)d;
            float s[K_CLASSES];
            for (int k = 0; k < K_CLASSES; k++) {
                s[k] = dot(W + (size_t)k * (size_t)d, xi, d) + b[k];
            }
            int pred = argmax3(s);
            int yi = y_train[i];
            if (pred != yi) {
                // Règle perceptron: on "pousse" la vraie classe vers le haut
                // et on "tire" la classe prédite vers le bas.
                float *w_y = W + (size_t)yi * (size_t)d;
                float *w_p = W + (size_t)pred * (size_t)d;
                for (int j = 0; j < d; j++) {
                    float v = lr * xi[j];
                    w_y[j] += v;
                    w_p[j] -= v;
                }
                b[yi] += lr;
                b[pred] -= lr;
                updates++;
            }
        }
        float acc_train = accuracy(W, b, X_train, y_train, n_train, d);
        printf("Epoch %d/%d: updates=%d, train_acc=%.3f\n", e + 1, epochs, updates, acc_train);
        if (updates == 0) {
            break;
        }
    }

    printf("Accuracy finale (train): %.3f\n", accuracy(W, b, X_train, y_train, n_train, d));
    printf("Accuracy finale (test) : %.3f\n", accuracy(W, b, X_test, y_test, n_test, d));

    free(W);
    free(X_train);
    free(X_test);
    free(y_train);
    free(y_test);
    return 0;
}
