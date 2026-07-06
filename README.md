## PA3-3IABD1-JAEM

Projet annuel Machine Learning — **classification d'images** (route sèche / route mouillée / route enneigée).

### Objectif

Construire un pipeline simple :
- **Prétraitement** (images → features / dataset \(X, y\))
- **Entraînement** des modèles (from scratch, implémentation perso en C)
- **Évaluation** (à compléter au fur et à mesure)

Modèles visés :
- **Modèle linéaire** (fait : testé sur datasets jouets `linear` et `xor`)
- **MLP** (à venir : doit réussir le XOR là où le linéaire échoue)
- **RBF** : en cours 
-  **SVM** (à venir)

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
├── built_datasets/                # variantes pré-construites (meta.json avec chemins relatifs)
│   ├── rgb_raw/                   # dry_road=0, wet_road=1, snowy_road=2
│   ├── rgb_norm/
│   ├── grayscale_raw/
│   ├── grayscale_norm/
│   ├── contours_raw/
│   └── contours_norm/
│
├── preprocessing/                 # Prétraitement / construction des datasets
│   ├── build_dataset.py           # images -> (X, y) train/test + export binaire C
│   ├── make_toy_dataset.py        # génère datasets/toy/linear.csv et xor.csv
│   └── utils.py                   # helpers
│
├── models/                        # Implémentations des modèles (en C)
│   ├── linear_model.c             # classifieur linéaire multi-classes (images, format binaire)
│   ├── linear_model_csv.c         # classifieur linéaire 2 classes (lit un CSV) + écrit poids.txt
│   ├── linear_modele_droite.c     # régression linéaire (exercice perso, hors sujet projet)
│   └── poids.txt                  # GÉNÉRÉ : poids w1 w2 b de la dernière exécution
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

> Remarque : les fichiers compilés en C (`models/linear_model_csv`, etc.) et `models/poids.txt`
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

`models/linear_model_csv.c` est un perceptron à 2 classes. Il lit un CSV, s'entraîne, affiche
l'accuracy par epoch, puis **écrit les poids `w1 w2 b` dans `models/poids.txt`** (pour le graphe).

Compilation (une fois) :

```bash
gcc models/linear_model_csv.c -o models/linear_model_csv
```

Entraînement :

```bash
# Cas séparable : doit converger vers accuracy = 1.00
./models/linear_model_csv datasets/toy/linear.csv

# Cas XOR : reste bloqué à accuracy = 0.50 (échec attendu)
./models/linear_model_csv datasets/toy/xor.csv
```

Arguments optionnels : `./models/linear_model_csv <fichier.csv> [epochs] [lr]`
(par défaut : 20 epochs, lr = 0.1).

#### 3) Tracer le graphe (points + droite de décision)

`visualization/plot_linear.py` lit les points (CSV) **et** les poids (`models/poids.txt`),
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
gcc models/linear_model.c -o models/linear_model -lm
```

Entraînement sur la variante NB normalisée (exemple) :

```bash
./models/linear_model \
  datasets/transformed/nb/normalisee/X_train.f32bin \
  datasets/transformed/nb/normalisee/y_train.i32bin
```

Arguments optionnels : `./models/linear_model <X.f32bin> <y.i32bin> [epochs] [lr]`
(par défaut : 30 epochs, lr = 0.1).

Remarques :
- `.npy` = format binaire NumPy (Python)
- `.f32bin` / `.i32bin` = format binaire lu par `models/linear_model.c`
- pas besoin d'un script d'export séparé : tout est généré par `build_dataset.py`