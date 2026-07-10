#include <stdint.h>
#define K_CLASSES 3

static float dot(const float *w, const float *x, int d) {
    float s = 0.0f;
    for (int j = 0; j < d; j++) {
        s += w[j] * x[j];
    }
    return s;
}

static int argmax3(const float s[K_CLASSES]){
    int best = 0;
    for (int k = 1; k < K_CLASSES; k++) {
        if (s[k] > s[best]){
            best = k;
        }
    }
    return best;
}

void fit(const float *X, const int *y, int n, int d, int epochs, float lr, float *W, float *b){
    for (int i = 0; i < K_CLASSES * d; i++) W[i] = 0.0f;
    for (int k = 0; k < K_CLASSES; k++) b[k] = 0.0f;
    
    for (int e = 0; e < epochs; e++){
        int updates = 0;
        for (int i = 0; i < n; i++){

            const float *xi = X + (long)i * d;
            float s[K_CLASSES];

            for (int k = 0; k < K_CLASSES; k++){
                s[k] = dot(W + (long)k * d, xi, d) + b[k];
            }

            int pred = argmax3(s);
            int yi = y[i];
            if (pred != yi) {

                float *w_y = W + (long)yi * d;
                float *w_p = W + (long)pred * d;

                for (int j = 0; j < d; j++){
                    float v = lr * xi[j];
                    w_y[j] += v;
                    w_p[j] -= v;
                }

                b[yi] += lr;
                b[pred] -= lr;
                updates++;
            }
        }
        if (updates == 0) break;
    }  
}

void predict(const float *X, int n, int d, const float *W, const float *b, int *out){
    for (int i = 0; i < n; i++){
        const float *xi = X + (long)i * d;
        float s[K_CLASSES];
        for (int k = 0; k < K_CLASSES; k++){
            s[k] = dot(W + (long)k * d, xi, d) + b[k];
        }
        out[i] = argmax3(s);
    }
}