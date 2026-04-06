## PA3-3IABD1-JAEM

Projet ML — classification d'images en 3 classes : **Hello Kitty**, **personnage Sanrio**, **autre**.

Les images sont stockées sur Google Drive (pas dans le dépôt). Le code tourne from scratch avec numpy.

---

### Installation

```bash
python -m venv .venv
source .venv/bin/activate    # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

---

### Arborescence

```
PA3-3IABD1-JAEM/
├── datasets/
│   ├── generate_linear.py      générer linear.csv
│   ├── generate_xor.py         générer xor.csv
│   ├── linear.csv
│   └── xor.csv
│
├── dataset_tools/
│   ├── drive_add_images.py     upload images vers Google Drive
│   ├── test_drive_add_images.py
│   └── README.md
│
├── preprocessing/
│   ├── build_dataset.py        images Drive -> X.npy / y.npy
│   ├── split_dataset.py        split train / test
│   └── utils.py
│
├── models/
│   ├── linear_model.py
│   └── mlp.py
│
└── training/
    ├── train_linear.py
    └── train_mlp.py
```

---

### Datasets simples

```bash
python datasets/generate_linear.py
python datasets/generate_xor.py
```

---

### Entraînement

```bash
python training/train_linear.py
python training/train_mlp.py
```

---

### Dataset réel (Google Drive)

Placer `credentials.json` dans `dataset_tools/`, puis :

```bash
python dataset_tools/drive_add_images.py --label hello_kitty --input /chemin/vers/images
python dataset_tools/drive_add_images.py --label sanrio_other --input /chemin/vers/images
python dataset_tools/drive_add_images.py --label other --input /chemin/vers/images
```

Construire le dataset local :

```bash
python preprocessing/build_dataset.py
python preprocessing/split_dataset.py --input datasets/transformed/rgb/normalisee --output datasets/splits/rgb
```
