#include <stdint.h>
#include <stdlib.h>

#define K_CLASSES 3

// accuracy sur un jeu de donnees avec les poids courants
static float accuracy_interne(const float *X, const int *y, int n, int d,
                              const float *W, const float *b)
{
    int corrects = 0;
    int i;
    int j;
    int k;

    for (i = 0; i < n; i = i + 1)
    {
        float scores[K_CLASSES];

        for (k = 0; k < K_CLASSES; k = k + 1)
        {
            float somme = 0.0f;
            for (j = 0; j < d; j = j + 1)
            {
                somme = somme + W[k * d + j] * X[i * d + j];
            }
            scores[k] = somme + b[k];
        }

        int classe_predite = 0;
        for (k = 1; k < K_CLASSES; k = k + 1)
        {
            if (scores[k] > scores[classe_predite])
            {
                classe_predite = k;
            }
        }

        if (classe_predite == y[i])
        {
            corrects = corrects + 1;
        }
    }

    return (float)corrects / (float)n;
}

void fit(const float *X, const int *y, int n, int d, int epochs, float lr, float *W, float *b)
{
    int i;
    int j;
    int k;
    int e;

    for (k = 0; k < K_CLASSES; k = k + 1)
    {
        for (j = 0; j < d; j = j + 1)
        {
            W[k * d + j] = 0.0f;
        }
    }

    for (k = 0; k < K_CLASSES; k = k + 1)
    {
        b[k] = 0.0f;
    }

    // On garde les meilleurs poids vus
    float *poche_W = (float *)calloc((size_t)K_CLASSES * (size_t)d, sizeof(float));
    float poche_b[K_CLASSES] = {0.0f, 0.0f, 0.0f};
    float meilleure_acc = -1.0f;

    for (e = 0; e < epochs; e = e + 1)
    {
        int nombre_de_corrections = 0;

        for (i = 0; i < n; i = i + 1)
        {
            float scores[K_CLASSES];

            for (k = 0; k < K_CLASSES; k = k + 1)
            {
                float somme = 0.0f;

                for (j = 0; j < d; j = j + 1)
                {
                    somme = somme + W[k * d + j] * X[i * d + j];
                }

                somme = somme + b[k];
                scores[k] = somme;
            }

            int classe_predite = 0;
            for (k = 1; k < K_CLASSES; k = k + 1)
            {
                if (scores[k] > scores[classe_predite])
                {
                    classe_predite = k;
                }
            }

            int vraie_classe = y[i];

            if (classe_predite != vraie_classe)
            {
                for (j = 0; j < d; j = j + 1)
                {
                    float valeur = lr * X[i * d + j];

                    W[vraie_classe * d + j] = W[vraie_classe * d + j] + valeur;

                    W[classe_predite * d + j] = W[classe_predite * d + j] - valeur;
                }

                b[vraie_classe] = b[vraie_classe] + lr;
                b[classe_predite] = b[classe_predite] - lr;

                nombre_de_corrections = nombre_de_corrections + 1;
            }
        }

        /* nouveau record train -> en poche */
        float acc_train = accuracy_interne(X, y, n, d, W, b);
        if (acc_train > meilleure_acc)
        {
            meilleure_acc = acc_train;
            for (j = 0; j < K_CLASSES * d; j = j + 1) { poche_W[j] = W[j]; }
            for (k = 0; k < K_CLASSES; k = k + 1) { poche_b[k] = b[k]; }
        }

        if (nombre_de_corrections == 0)
        {
            break;
        }
    }

    for (j = 0; j < K_CLASSES * d; j = j + 1) { W[j] = poche_W[j]; }
    for (k = 0; k < K_CLASSES; k = k + 1) { b[k] = poche_b[k]; }
    free(poche_W);
}

void predict(const float *X, int n, int d, const float *W, const float *b, int *out) {
    int i;
    int j;
    int k;

    for (i = 0; i < n; i = i + 1)
    {
        float scores[K_CLASSES];

        for (k = 0; k < K_CLASSES; k = k + 1)
        {
            float somme = 0.0f;

            for (j = 0; j < d; j = j + 1)
            {
                somme = somme + W[k * d + j] * X[i * d + j];
            }

            somme = somme + b[k];
            scores[k] = somme;
        }

        int classe_predite = 0;
        for (k = 1; k < K_CLASSES; k = k + 1)
        {
            if (scores[k] > scores[classe_predite])
            {
                classe_predite = k;
            }
        }

        out[i] = classe_predite;
    }
}                  