# Prototype Gradio — état de la route

Ce dossier ajoute une interface Gradio minimale au projet. L’utilisateur charge une image, choisit l’un des trois modèles implémentés, puis obtient une classe : route sèche, mouillée ou enneigée.

## Principe retenu

- Une seule représentation est utilisée par les trois modèles : **niveaux de gris normalisés**.
- Les données d’entraînement viennent de `datasets/transformed/nb/normalisee`.
- Si des paramètres sauvegardés existent dans `models/cache_gradio`, ils sont rechargés sans charger la bibliothèque native.
- Sinon, le modèle sélectionné est entraîné lors de sa première utilisation, puis ses paramètres sont sauvegardés au format `.npz`.
- Le RBF peut prendre davantage de temps lors du premier lancement.

## Informations affichées

- **Modèle linéaire** : scores bruts `W × x + b` pour les trois classes.
- **MLP** : probabilités softmax internes calculées par la couche de sortie. Elles ne sont pas calibrées.
- **RBF** : scores bruts calculés à partir des influences gaussiennes et des poids de sortie.

Les scores du modèle linéaire et du RBF ne sont pas des probabilités. Dans les trois cas, la classe dont la sortie est la plus élevée devient la prédiction.

## Installation

Depuis la racine du dépôt :

```bash
python -m venv .venv
```

Sous Windows avec Git Bash :

```bash
source .venv/Scripts/activate
```

Installer les dépendances :

```bash
python -m pip install -r requirements.txt
python -m pip install -r site_gradio/requirements.txt
```

Construire le jeu de données s’il n’existe pas encore :

```bash
python preprocessing/build_dataset.py --skip-drive
```

## Compilation multiplateforme

Le script choisit automatiquement `.dll` sous Windows, `.so` sous Linux et `.dylib` sous macOS :

```bash
python site_gradio/build_libs.py
```

Sous Windows, `gcc` et `g++` doivent être accessibles dans le `PATH`. Le RBF nécessite également `models/rbf/eigen-5.0.0`.

## Lancement

```bash
python site_gradio/app.py
```

Ouvrir ensuite l’adresse locale affichée par Gradio, généralement `http://127.0.0.1:7860`.

## Correction Windows du RBF

L’ancienne version cherchait systématiquement `models/rbf/rbf_lib.so`. Windows charge une bibliothèque `.dll`, pas une bibliothèque Linux `.so`. La version corrigée :

1. choisit l’extension selon le système ;
2. compile `rbf_lib.dll` sous Windows ;
3. lie statiquement les bibliothèques d’exécution GCC/G++ pour limiter les DLL manquantes ;
4. ajoute le dossier de la bibliothèque au chemin de recherche de Windows avant le chargement.

## Limites

- Il s’agit d’une démonstration locale, sans authentification ni base de données.
- Les probabilités softmax du MLP ne sont pas calibrées.
- Les sorties du modèle linéaire et du RBF sont uniquement des scores bruts.
- La qualité dépend directement du jeu d’entraînement et des paramètres fixés dans `app.py`.
- Pour un déploiement réel, il est préférable de produire les paramètres hors ligne et de ne conserver que l’inférence dans l’application.
