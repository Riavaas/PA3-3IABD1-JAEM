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
│       ├── sanrio_other/
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


