## PA3-3IABD1-JAEM

Projet annuel Machine Learning — **classification d’images** (Hello Kitty / Fake Hello Kitty / Other).

### Objectif

Construire un pipeline simple :
- **Prétraitement** (images → features / dataset \(X, y\))
- **Entraînement** des modèles (from scratch)
- **Évaluation** (à compléter au fur et à mesure)

Modèles visés :
- **Modèle linéaire** (en cours)
- **MLP** (à venir)
- **RBF / SVM** (à venir)

### Arborescence

```text
PA3-3IABD1-JAEM/
├── README.md
├── .gitignore
├── requirements.txt
│
├── dataset_tools/                 # Scripts Google Drive (ajout d’images)
│   ├── drive_add_images.py
│   ├── test_drive_add_images.py
│   └── README.md
│
├── datasets/                      # Données (ignorées par git)
│   └── raw/
│       ├── hello_kitty/
│       ├── fake_hello_kitty/
│       └── other/
│
├── preprocessing/                 # Prétraitement / construction du dataset
│   ├── build_dataset.py           # images -> (X, y)
│   ├── split_dataset.py           # split train / test
│   └── utils.py                   # helpers
│
├── models/                        # Implémentations des modèles
│   └── linear_model.py
│
└── training/                      # Scripts d’entraînement
    └── train_linear.py
```

### Installation

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Construction du dataset (`preprocessing/build_dataset.py`)

Le script `preprocessing/build_dataset.py` construit automatiquement le dataset utilisable pour l'entraînement.

Ce qu'il fait :
- (par défaut) télécharge les images depuis Google Drive (`dataset/hello_kitty`, `dataset/fake_hello_kitty`, `dataset/other`)
- conserve une taille d'image unique (les images d'une autre taille sont ignorées avec un warning)
- génère les features pour 3 variantes : `rgb`, `nb` (niveaux de gris), `contours`
- exporte chaque variante en version `normalisee` et `non_normalisee`
- sauvegarde `X.npy` (features) et `y.npy` (labels)

Commande par défaut (avec téléchargement Drive) :

```bash
python3 preprocessing/build_dataset.py
```

Mode local (sans téléchargement Drive) :

```bash
python3 preprocessing/build_dataset.py --skip-drive
```

Tu peux aussi injecter des sources locales :

```bash
python3 preprocessing/build_dataset.py \
  --skip-drive \
  --source-vrai /chemin/vers/hello_kitty \
  --source-faux /chemin/vers/fake_hello_kitty \
  --source-other /chemin/vers/other
```

Emplacements de stockage :
- `datasets/raw/` : images brutes par classe
- `datasets/transformed/` : datasets prêts pour ML
  - `rgb/normalisee/X.npy`, `rgb/normalisee/y.npy`
  - `rgb/non_normalisee/X.npy`, `rgb/non_normalisee/y.npy`
  - `nb/normalisee/...`, `nb/non_normalisee/...`
  - `contours/normalisee/...`, `contours/non_normalisee/...`

Remarques :
- `.npy` est un format binaire NumPy 
- Les dossiers `datasets/` sont ignorés par Git (données volumineuses).


