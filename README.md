## PA3-3IABD1-JAEM

Projet annuel Machine Learning — **classification d'images** (route sèche / route mouillée / route enneigée).

### Objectif

Construire un pipeline simple :
- **Prétraitement** (images → features / dataset \(X, y\))
- **Entraînement** des modèles (from scratch, implémentation perso en C)
- **Évaluation** (à compléter au fur et à mesure)

Modèles visés :
- **Modèle linéaire** (fait : testé sur datasets jouets `linear`/`xor`, appliqué aux images, bibliothèque dynamique avec pocket)
- **MLP** (fait : réussit XOR là où le linéaire échoue, appliqué aux images, bibliothèque dynamique)
- **RBF** (fait : version naïve validée sur points jouets + version kmeans sur les images, deux modes d'apprentissage — pinv et rosenblatt+pocket+shuffle — bibliothèque dynamique)
- **SVM** (à venir)

Livrables complémentaires :
- **`site_gradio/`** : application cliente (upload d'une photo, choix du modèle, prédiction), qui charge les 3 modèles via leurs bibliothèques dynamiques avec sauvegarde/chargement des poids entraînés.

### Arborescence (état actuel)

```text
PA3-3IABD1-JAEM/
├── README.md
├── .gitignore
├── requirements.txt
│
├── dataset_tools/                 # Scripts Google Drive (ajout d'images)
│   ├── drive_add_images.py
│   ├── test_drive_add_images.py
│   └── README.md
│
├── datasets/                      # Données (ignorées par git, sauf toy/)
│   ├── raw/                       # images brutes par classe
│   │   ├── dry_road/
│   │   ├── wet_road/
│   │   └── snowy_road/
│   ├── transformed/               # features prêtes pour le ML (rgb / nb / contours)
│   └── toy/                       # petits datasets de test (générés)
│       ├── linear.csv             # 2 groupes séparables par une droite
│       └── xor.csv                # 4 points en croix (non séparables)
│
├── preprocessing/                 # Prétraitement / construction des datasets
│   ├── build_dataset.py           # images -> (X, y) train/test + export binaire C
│   ├── make_toy_dataset.py        # génère datasets/toy/linear.csv et xor.csv
│   └── utils.py                   # helpers
│
├── models/                        # Implémentations des modèles (en C / C++)
│   ├── lineaire/                  # modèles linéaires + fichiers liés
│   │   ├── linear_model.c         # perceptron multi-classes (images, format binaire)
│   │   ├── linear_model_csv.c     # perceptron 2 classes (lit un CSV) + écrit poids.txt
│   │   ├── linear_model_lib.c     # bibliothèque dynamique (fit/predict, avec pocket)
│   │   ├── linear_modele_droite.c # régression linéaire (exercice perso, hors sujet projet)
│   │   ├── notebook_linear.ipynb  # notebook : lance le C (images) et trace les graphes
│   │   ├── test_linear_lib.ipynb  # notebook : teste la bibliothèque dynamique (ctypes)
│   │   ├── poids.txt              # GÉNÉRÉ : poids w1 w2 b de la dernière exécution
│   │   └── test_points.txt
│   │
│   ├── mlp/                       # Perceptron Multi-Couches (PMC)
│   │   ├── mlp_csv.c               # version jouet 2 classes (lit un CSV)
│   │   ├── mlp.c                   # version images (K=3, softmax)
│   │   ├── mlp_lib.c               # bibliothèque dynamique (entrainer/predire_batch)
│   │   ├── notebook_mlp.ipynb      # notebook : lance le C et trace les graphes
│   │   ├── test_mlp_lib.ipynb      # notebook : teste la bibliothèque dynamique (ctypes)
│   │   ├── JOURNAL.md              # notes d'expérimentation (bugs, stabilité multi-graines)
│   │   └── poids_mlp.txt           # GÉNÉRÉ : poids de la dernière exécution
│   │
│   ├── rbf/                        # RBF Network (C++)
│   │   ├── rbf_simple.cpp          # version pédago : 6 points hardcodés (naïf + kmeans)
│   │   ├── rbf.cpp                 # version images, modes pinv et rosenblatt+pocket+shuffle
│   │   ├── rbf_lib.cpp             # bibliothèque dynamique (entrainer pinv + entrainer_rosenblatt)
│   │   ├── notebook_rbf.ipynb      # notebook : lance le C++ et trace les graphes
│   │   ├── test_rbf_lib.ipynb      # notebook : teste la bibliothèque dynamique (ctypes)
│   │   ├── suivi_resultats.md      # trace des runs RBF (gamma, K, seeds, résultats)
│   │   └── eigen-5.0.0/            # librairie Eigen (IGNORÉE par git, voir Partie RBF)
│   │
│   └── cache_gradio/                # GÉNÉRÉ : poids sauvegardés (.npz) pour le site
│
├── site_gradio/                    # Application cliente (Gradio)
│   ├── app.py                      # charge les 3 modèles via ctypes, prédiction sur upload
│   ├── build_libs.py / build_libs.sh # compile les 3 bibliothèques dynamiques
│   ├── requirements.txt
│   └── README.md                   # installation, fonctionnement, limites
│
├── visualization/                 # Scripts qui produisent des graphes
│   ├── plot_linear.py             # points + droite de décision (lit poids.txt)
│   ├── visualize_transformed_rgb.py
│   ├── visualize_transformed_nb.py
│   └── visualize_transformed_contours.py
│
└── training/                      # Scripts d'entraînement (à compléter)
    └── train_linear.py
```

> Remarque : les fichiers compilés en C (`models/lineaire/linear_model_csv`, etc.) et `models/lineaire/poids.txt`
> sont régénérés ; pas besoin de les versionner.

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```


> ```bash
> rm -rf .venv
> python3 -m venv .venv
> source .venv/bin/activate
> python3 -m pip install -r requirements.txt
> ```

---

### Partie 1 — Datasets jouets + modèle linéaire en C + graphes

Cette partie prouve que le modèle linéaire fonctionne sur un cas facile (`linear`) et échoue sur
un cas non séparable (`xor`), comme demandé dans le sujet.

#### 1) Générer les datasets de test

Le script `preprocessing/make_toy_dataset.py` écrit `datasets/toy/linear.csv` et `datasets/toy/xor.csv`.
Il utilise un chemin relatif (`../datasets/toy`), donc on le lance **depuis le dossier `preprocessing/`** :

```bash
cd preprocessing
python3 make_toy_dataset.py
cd ..
```

Contenu attendu :
- `linear.csv` : 6 points, classes (0/1) séparables par une droite.
- `xor.csv` : 4 points (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0.

#### 2) Entraîner le modèle linéaire (en C)

`models/lineaire/linear_model_csv.c` est un perceptron à 2 classes. Il lit un CSV, s'entraîne, affiche
l'accuracy par epoch, puis **écrit les poids `w1 w2 b` dans `models/lineaire/poids.txt`** (pour le graphe).

Compilation (une fois) :

```bash
gcc models/lineaire/linear_model_csv.c -o models/lineaire/linear_model_csv
```

Entraînement :

```bash
# Cas séparable : doit converger vers accuracy = 1.00
./models/lineaire/linear_model_csv datasets/toy/linear.csv

# Cas XOR : reste bloqué à accuracy = 0.50 (échec attendu)
./models/lineaire/linear_model_csv datasets/toy/xor.csv
```

Arguments optionnels : `./models/lineaire/linear_model_csv <fichier.csv> [epochs] [lr]`
(par défaut : 20 epochs, lr = 0.1).

#### 3) Tracer le graphe (points + droite de décision)

`visualization/plot_linear.py` lit les points (CSV) **et** les poids (`models/lineaire/poids.txt`),
puis dessine les points colorés par classe et la droite de décision.

```bash
python3 visualization/plot_linear.py datasets/toy/linear.csv
python3 visualization/plot_linear.py datasets/toy/xor.csv
```

> ⚠️ Ordre important : `poids.txt` est réécrit à chaque exécution du C. Il faut donc lancer
> le programme C **puis** le script Python **sur le même CSV, à la suite**, sinon on trace
> les points d'un dataset avec la droite d'un autre.

Résultat attendu :
- `linear.csv` → bleus en bas à gauche, oranges en haut à droite, droite en diagonale entre les deux.
- `xor.csv` → droite verticale qui laisse les couleurs mélangées des deux côtés (échec).

L'image est sauvegardée dans `plot_linear.png`.

---

### Construction du dataset images (`preprocessing/build_dataset.py`)

Le script `preprocessing/build_dataset.py` construit automatiquement le dataset utilisable pour l'entraînement.

Ce qu'il fait :
- (par défaut) télécharge les images depuis Google Drive (`dataset_routes/dry_road`, `dataset_routes/wet_road`, `dataset_routes/snowy_road`)
- conserve une taille d'image unique (les images d'une autre taille sont ignorées avec un warning)
- génère les features pour 3 variantes : `rgb`, `nb` (niveaux de gris), `contours`
- exporte chaque variante en version `normalisee` et `non_normalisee`
- fait un split train/test **80/20** (reproductible, `random.seed(67)`)
- sauvegarde **Python** (`.npy`) **et** **C** (`.f32bin` / `.i32bin`) dans le même dossier

#### Commandes

Activer l'environnement (à faire une fois par session) :

```bash
cd PA3-3IABD1-JAEM
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Construire le dataset (recommandé si les images sont déjà dans `datasets/raw/`) :

```bash
python3 preprocessing/build_dataset.py --skip-drive
```

Construire avec téléchargement Google Drive (plus long) :

```bash
python3 preprocessing/build_dataset.py
```

Injecter des sources locales en plus :

```bash
python3 preprocessing/build_dataset.py \
  --skip-drive \
  --source-dry-road /chemin/vers/dry_road \
  --source-wet-road /chemin/vers/wet_road \
  --source-snowy-road /chemin/vers/snowy_road
```

#### Fichiers générés

Emplacements :
- `datasets/raw/` : images brutes par classe
- `datasets/transformed/` : datasets prêts pour ML

Dans chaque dossier (`rgb/nb/contours` × `normalisee/non_normalisee`) :

| Fichier | Usage |
|---------|--------|
| `X_train.npy`, `y_train.npy` | Python (notebook) |
| `X_test.npy`, `y_test.npy` | Python (évaluation) |
| `X_train.f32bin`, `y_train.i32bin` | C (entraînement) |
| `X_test.f32bin`, `y_test.i32bin` | C (test) |

Labels : `dry_road=0`, `wet_road=1`, `snowy_road=2`.

> Les dossiers `datasets/raw/` et `datasets/transformed/` sont **ignorés par Git** (fichiers trop lourds, >100 Mo par fichier). Chaque membre du groupe régénère le dataset en local avec `build_dataset.py`.

#### Entraîner le modèle linéaire sur les images (C)

Compilation :

```bash
gcc models/lineaire/linear_model.c -o models/lineaire/linear_model -lm
```

Entraînement sur la variante NB normalisée (exemple, chemins par défaut) :

```bash
./models/lineaire/linear_model
```

Avec chemins explicites (train + test) :

```bash
./models/lineaire/linear_model \
  datasets/transformed/nb/normalisee/X_train.f32bin \
  datasets/transformed/nb/normalisee/y_train.i32bin \
  datasets/transformed/nb/normalisee/X_test.f32bin \
  datasets/transformed/nb/normalisee/y_test.i32bin
```

Arguments optionnels :
`./models/lineaire/linear_model <X_train> <y_train> <X_test> <y_test> [epochs] [lr]`
(par défaut : chemins NB normalisée ci-dessus, 30 epochs, lr = 0.1).

Le split train/test est fait en Python (`build_dataset.py`) ; le C entraîne sur le train
et affiche à chaque epoch une ligne `epoch <e> train <acc> test <acc>`, puis une matrice
de confusion 3×3 sur le test (format lisible par le notebook).

Remarques :
- `.npy` = format binaire NumPy (Python)
- `.f32bin` / `.i32bin` = format binaire lu par `models/lineaire/linear_model.c`
- pas besoin d'un script d'export séparé : tout est généré par `build_dataset.py`

#### Notebook interactif (entraîner + visualiser facilement)

`models/lineaire/notebook_linear.ipynb` sert à lancer le modèle linéaire (images, K=3)
et à afficher les résultats. Il ne calcule rien lui-même : il compile et exécute
`linear_model.c`, récupère sa sortie et trace les graphes avec Matplotlib.

Prérequis : dataset construit (`build_dataset.py`) et `gcc` disponible. Lancer Jupyter
puis ouvrir le notebook :

```bash
source .venv/bin/activate
python3 -m pip install notebook   # si Jupyter n'est pas déjà installé
jupyter notebook models/lineaire/notebook_linear.ipynb
```

Variables à changer (cellule 1, en haut) :

| Variable | Rôle | Valeurs |
|----------|------|---------|
| `variante` | type de features | `"rgb"`, `"nb"`, `"contours"` |
| `normalisation` | pixels bruts ou 0–1 | `"normalisee"`, `"non_normalisee"` |
| `epochs` | nombre de passages sur le train | ex. `30` |
| `lr` | vitesse d'apprentissage | ex. `0.1` |

Ce qu'il faut regarder :
- **Courbe train vs test** : si le train monte et que le test stagne, c'est du **surapprentissage**.
- **Matrice de confusion (test)** : montre quelles classes de routes se confondent.
- **Bar chart des 6 variantes** : compare l'accuracy de test finale.

Résultat attendu : accuracy de test autour de **0.40**, à peine au-dessus du hasard (~0.33 pour
3 classes) → modèle linéaire trop faible pour ces images, ce qui justifie de passer au MLP.

---

### Partie MLP — Perceptron Multi-Couches (C)

Le MLP ajoute une couche cachée (activation sigmoïde) entre l'entrée et la sortie (softmax,
K=3 classes), ce qui lui permet d'apprendre des frontières non linéaires — contrairement au
modèle linéaire, il réussit XOR. Rétropropagation complète (couche cachée corrigée avec
l'**ancien** poids de sortie, avant sa propre mise à jour — piège d'ordre documenté dans
`models/mlp/JOURNAL.md`).

Fichiers :
- `models/mlp/mlp_csv.c` : version jouet 2 classes (lit un CSV), validée sur `linear.csv` et `xor.csv` (100%).
- `models/mlp/mlp.c` : version images (mêmes fichiers `.f32bin`/`.i32bin` que `linear_model.c`), K=3, softmax.
- `models/mlp/mlp_lib.c` : bibliothèque dynamique (`entrainer`/`predire_batch`), utilisée par `site_gradio/`.
- `models/mlp/notebook_mlp.ipynb`, `models/mlp/test_mlp_lib.ipynb` : notebooks interactifs.
- `models/mlp/JOURNAL.md` : notes d'expérimentation (bugs rencontrés, stabilité multi-graines).

#### Entraîner sur les images (C)

```bash
gcc -O2 models/mlp/mlp.c -o models/mlp/mlp -lm
./models/mlp/mlp \
  datasets/transformed/rgb/normalisee/X_train.f32bin \
  datasets/transformed/rgb/normalisee/y_train.i32bin \
  datasets/transformed/rgb/normalisee/X_test.f32bin \
  datasets/transformed/rgb/normalisee/y_test.i32bin \
  32 30 0.001 67
```

Arguments : `./models/mlp/mlp <X_train> <y_train> <X_test> <y_test> [H] [epochs] [lr] [seed]`
(par défaut : NB normalisée, H=32, epochs=30, lr=0.001, seed=67).

**Important** : n'utiliser que des variantes **normalisées**. Sur des pixels bruts non normalisés,
le MLP s'effondre totalement (sature la sigmoïde dès l'initialisation, prédit toujours la même
classe) — contrairement au modèle linéaire qui dégrade juste. Meilleur résultat obtenu :
`rgb/normalisee`, H=32, accuracy test ≈ 0.618, validé sur 3 graines (67/42/96) avec une variance
faible (0.619 ± 0.004, voir `JOURNAL.md`).

Sortie : une ligne `epoch <e> train <acc> test <acc>` par epoch, puis une matrice de confusion
3×3 sur le test, et les poids sont sauvegardés dans `models/mlp/poids_mlp.txt`.

---

### Partie RBF — Radial Basis Function Network (C++)

Le RBF remplace les hyperplans du modèle linéaire par des « influences » gaussiennes :
chaque centre émet une influence `phi(x, c) = e^(-gamma * ||x - c||²)` qui décroît avec
la distance. Pipeline :

1. `kmeans` choisit K centres représentatifs du train (chaque centre = moyenne de son groupe de points)
2. on construit la matrice `phi` (N × K) : influence de chaque point sur chaque centre
3. apprentissage des poids de sortie, **deux modes** :
   - **`rosenblatt`** (défaut) : règle du perceptron multi-classes appliquée dans l'espace
     des `phi` (même règle que `linear_model.c` : si la prédiction est fausse, on pousse
     les poids de la vraie classe et on tire ceux de la classe prédite, avec biais).
     Itératif : hyperparamètres `epochs` et `lr`, arrêt anticipé si plus aucune erreur train.
     **Pocket algorithm** : comme les données ne sont pas forcément séparables dans l'espace
     des phi, le perceptron oscille ; on garde donc en mémoire les meilleurs poids rencontrés
     (selon l'accuracy **train**, jamais le test) et ce sont eux qui sont rendus à la fin
     (ligne `pocket epoch <e> train <acc>` dans la sortie).
     **Shuffle** : l'ordre de passage des exemples est remélangé à chaque epoch
     (Fisher-Yates sur `rand()`, donc reproductible via la seed) — un ordre figé crée des
     cycles de corrections qui se répètent et alimentent l'oscillation.
   - **`pinv`** : les poids sont résolus d'un coup, `W = phi⁺ · Y` (pseudo-inverse par SVD,
     Y en one-hot, solution des moindres carrés). Pas d'epochs ni de lr. Gardé pour
     comparaison avec la version Rosenblatt.
4. prédiction : `argmax_k b[k] + Σ_c W[k][c] * phi_c(x)` (b = 0 en mode pinv)

Fichiers :
- `models/rbf/rbf_simple.cpp` : version pédagogique sur 6 points hardcodés — RBF **naïf**
  (centres = tous les points) puis RBF **kmeans**. Sert à prouver la justesse de l'implémentation.
- `models/rbf/rbf.cpp` : version images (mêmes fichiers `.f32bin`/`.i32bin` et même format de
  sortie que `linear_model.c`).
- `models/rbf/notebook_rbf.ipynb` : notebook interactif.

#### 0) Dépendance : Eigen (une fois par machine)

La pseudo-inverse utilise **Eigen 5.0.0** (librairie de matrices, headers only, rien à installer).
Le dossier `models/rbf/eigen-5.0.0/` est **ignoré par git** (~3000 fichiers de headers) :

```bash
curl -L https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz | tar xz -C models/rbf/
```

(ou télécharger le zip sur https://eigen.tuxfamily.org et le dézipper dans `models/rbf/`,
le dossier doit s'appeler exactement `eigen-5.0.0`).

#### 1) Vérifier la justesse (points jouets)

```bash
g++ -O2 -std=c++17 models/rbf/rbf_simple.cpp -o models/rbf/rbf_simple -I models/rbf
./models/rbf/rbf_simple
```

Résultat attendu :
- **naïf** : accuracy = 1.00 — avec autant de centres que de points, le système `W = phi⁻¹·Y`
  interpole exactement les données (normal, c'est la propriété du RBF naïf)
- **kmeans (4 centres pour 6 points)** : accuracy ≈ 0.83 — moins de centres = modèle plus
  simple, on perd un point (c'est le compromis complexité/généralisation)

#### 2) Entraîner sur les images

```bash
g++ -O2 -std=c++17 models/rbf/rbf.cpp -o models/rbf/rbf -I models/rbf
```

Chemins par défaut (NB normalisée, gamma=0.01, 100 centres, seed=42, mode rosenblatt,
100 epochs, lr=0.1) :

```bash
./models/rbf/rbf
```

Avec tout explicite :

```bash
# mode rosenblatt (défaut)
./models/rbf/rbf \
  datasets/transformed/nb/normalisee/X_train.f32bin \
  datasets/transformed/nb/normalisee/y_train.i32bin \
  datasets/transformed/nb/normalisee/X_test.f32bin \
  datasets/transformed/nb/normalisee/y_test.i32bin \
  0.01 100 42 rosenblatt 100 0.1

# mode pseudo-inverse (pour comparaison)
./models/rbf/rbf \
  datasets/transformed/nb/normalisee/X_train.f32bin \
  datasets/transformed/nb/normalisee/y_train.i32bin \
  datasets/transformed/nb/normalisee/X_test.f32bin \
  datasets/transformed/nb/normalisee/y_test.i32bin \
  0.01 100 42 pinv
```

Arguments : `./models/rbf/rbf <X_train> <y_train> <X_test> <y_test> [gamma] [nb_centres] [seed] [mode] [epochs] [lr]`
(`mode` = `rosenblatt` ou `pinv` ; `epochs` et `lr` ne servent qu'en mode rosenblatt).

Sortie : en mode rosenblatt, une ligne `epoch <e> train <acc> test <acc>` par epoch
(comme le linéaire), puis dans les deux modes `acc train <x>`, `acc test <x>` et la
matrice de confusion 3×3 sur le test (même format que le linéaire, lisible par le notebook).

#### 3) Notebook interactif

```bash
source .venv/bin/activate
jupyter notebook models/rbf/notebook_rbf.ipynb
```

Variables à changer (cellule 1, en haut) :

| Variable | Rôle | Valeurs |
|----------|------|---------|
| `variante` | type de features | `"rgb"`, `"nb"`, `"contours"` |
| `normalisation` | pixels bruts ou 0–1 | `"normalisee"`, `"non_normalisee"` |
| `gamma` | largeur des gaussiennes | ex. `0.01` |
| `nb_centres` | K du kmeans (complexité du modèle) | ex. `100` |
| `seed` | graine de l'init aléatoire du kmeans | ex. `42` |
| `mode` | méthode d'apprentissage des poids | `"rosenblatt"`, `"pinv"` |
| `epochs` | passages sur le train (rosenblatt) | ex. `100` |
| `lr` | vitesse d'apprentissage (rosenblatt) | ex. `0.1` |

Ce qu'il faut regarder :
- **Courbe train/test par epoch** (mode rosenblatt) : si le train monte et que le test
  stagne, c'est du **surapprentissage** (comme pour le linéaire).
- **Courbe impact de gamma** : gamma trop grand → gaussiennes très étroites → apprentissage
  « par cœur » (train haut, test bas = **surapprentissage**) ; gamma trop petit → toutes les
  influences se valent → **sous-apprentissage**.
- **Courbe impact de nb_centres** : plus de centres = modèle plus complexe (mêmes phénomènes).
- **Matrice de confusion (test)** et **bar chart des 6 variantes** : comme pour le linéaire.
- **Comparaison rosenblatt vs pinv** (dernière cellule) : mêmes centres kmeans (même seed),
  seule la méthode d'apprentissage des poids change.

Remarques :
- en mode **rosenblatt**, le RBF a des epochs et un lr (perceptron dans l'espace des phi) ;
  en mode **pinv**, un run = un entraînement complet en une passe (kmeans + pseudo-inverse)
- perf : `rgb` (d=49152) est ~10× plus lent que `nb` (d=4096) → régler gamma/nb_centres
  sur `nb` d'abord, puis lancer la comparaison des variantes
- les binaires compilés (`models/rbf/rbf`, `models/rbf/rbf_simple`) sont ignorés par git

---

### Bibliothèques dynamiques et application cliente (`site_gradio/`)

Chaque modèle (linéaire, MLP, RBF) a une version bibliothèque dynamique en plus de sa version
exécutable, appelable depuis Python via `ctypes` (données échangées directement en mémoire,
sans fichier intermédiaire) :

| Modèle | Source | Fonctions exposées |
|--------|--------|---------------------|
| Linéaire | `models/lineaire/linear_model_lib.c` | `fit(...)`, `predict(...)` — avec pocket |
| MLP | `models/mlp/mlp_lib.c` | `entrainer(...)`, `predire_batch(...)` |
| RBF | `models/rbf/rbf_lib.cpp` | `entrainer(...)` (pinv), `entrainer_rosenblatt(...)` (nouveau, pas encore branché sur le site) |

Compiler les 3 bibliothèques (choisit automatiquement `.dll`/`.so`/`.dylib` selon l'OS) :

```bash
python site_gradio/build_libs.py
```

Sous Windows, `gcc`/`g++` doivent être dans le `PATH`. Le RBF nécessite en plus `models/rbf/eigen-5.0.0/` (voir plus haut).

Chaque bibliothèque a son propre notebook de test isolé (`test_linear_lib.ipynb`, `test_mlp_lib.ipynb`, `test_rbf_lib.ipynb`), à côté du modèle correspondant.

#### Lancer l'application cliente

```bash
python -m pip install -r requirements.txt
python -m pip install -r site_gradio/requirements.txt
python preprocessing/build_dataset.py --skip-drive   # si le dataset transformé n'existe pas
python site_gradio/build_libs.py
python site_gradio/app.py
```

Puis ouvrir `http://127.0.0.1:7860`. On charge une photo, on choisit un modèle, on obtient une prédiction.

Au premier lancement, chaque modèle est entraîné une fois (via sa bibliothèque) puis ses poids
sont sauvegardés dans `models/cache_gradio/*.npz` — au lancement suivant, ils sont rechargés
directement, sans ré-entraînement (sauvegarde/chargement demandé par le syllabus). **Si une
bibliothèque est mise à jour (ex. ajout du pocket), il faut supprimer le fichier `.npz`
correspondant dans `models/cache_gradio/` pour forcer un nouvel entraînement**, sinon le site
continue de servir d'anciens poids.

Limites connues (détaillées dans `site_gradio/README.md`) : démonstration locale uniquement,
scores du linéaire/RBF non calibrés (pas des probabilités), une seule variante de prétraitement
utilisée (`nb/normalisee`), le RBF utilise encore son ancienne fonction (`entrainer`, pinv) —
`entrainer_rosenblatt` existe dans la bibliothèque mais n'est pas encore branchée côté site.