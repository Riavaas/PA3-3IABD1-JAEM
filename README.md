# PA3-3IABD1-JAEM

Projet annuel Machine Learning — structure minimale.

## Arborescence

```
PA3-3IABD1-JAEM/
├── README.md
├── .gitignore
├── requirements.txt
├── ml-lib/           # Implémentation C++ des modèles
│   ├── linear_model.cpp
│   ├── linear_model.h
│   ├── mlp.cpp
│   └── mlp.h
├── python/           # Scripts de test et visualisation
│   ├── generate_datasets.py
│   ├── test_linear.py
│   ├── test_mlp.py
│   └── plot.py
├── datasets/         # Jeux de données de test
│   ├── linear.csv
│   └── xor.csv
└── dataset_tools/     # Outils dataset Google Drive (Hello Kitty)
    ├── drive_add_images.py
    └── test_drive_add_images.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

- **ml-lib** : compiler et utiliser les binaires C++ (à compléter).
- **python/** : génération de données, tests des modèles, plots.
- **dataset_tools/** : alimentation du dataset images sur Google Drive (OAuth).  
  Voir `dataset_tools/README.md` ou le docstring de `drive_add_images.py` pour la config.

## Tests

```bash
# Tests dataset_tools (depuis la racine du projet)
cd dataset_tools && pytest test_drive_add_images.py -v
```
