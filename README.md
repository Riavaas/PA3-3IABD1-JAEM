# PA3-3IABD1-JAEM

Projet annuel Machine Learning — Classification d’images  
(Hello Kitty / Fake Hello Kitty / Other)

---

## 🧠 Objectif

Implémenter plusieurs modèles de Machine Learning (from scratch) et les appliquer à un dataset d’images :

- Modèle linéaire (en cours)
- Perceptron multi-couches (à venir)
- RBF / SVM (à venir)

Le projet inclut :
- Prétraitement des images
- Entraînement des modèles
- Analyse des performances
- (Plus tard) API + application

---

## 📁 Arborescence actuelle

PA3-3IABD1-JAEM/
├── README.md
├── .gitignore
├── requirements.txt
│
├── dataset_tools/            # Scripts Google Drive (upload images)
│   ├── drive_add_images.py
│   ├── test_drive_add_images.py
│   └── README.md
│
├── datasets/                # Données (ignorées par git)
│   └── raw/
│       ├── hello_kitty/
│       ├── fake_hello_kitty/
│       └── other/
│
├── preprocessing/           # Transformation des images
│   ├── build_dataset.py     # images → vecteurs (X, y)
│   ├── split_dataset.py     # train / test split
│   └── utils.py             # fonctions utilitaires
│
├── models/                  # Implémentation des modèles
│   └── linear_model.py
│
├── training/                # Scripts d’entraînement
│   └── train_linear.py


---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


