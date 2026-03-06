# dataset_tools

Outils pour alimenter le dataset images (Google Drive) : Hello Kitty / Fake Hello Kitty / Other.

- **drive_add_images.py** : ajoute des images dans `Mon Drive/dataset/{hello_kitty|fake_hello_kitty|other}` avec normalisation (128×128, JPG), déduplication (md5 + phash) et log CSV.
- **test_drive_add_images.py** : tests pytest (mock, pas d’appel réel à l’API).

## Configuration

1. Mettre `credentials.json` (OAuth Desktop) dans ce dossier (ou passer `--credentials`).
2. Lancer une fois : une fenêtre s’ouvrira pour autoriser le compte Google → `token.json` sera créé.

Ne jamais committer `credentials.json` ni `token.json`.

## Utilisation

Depuis la **racine du projet** :

```bash
python dataset_tools/drive_add_images.py --label hello_kitty --input /chemin/vers/images
```

Ou depuis **dataset_tools/** (avec `credentials.json` et `token.json` dans ce dossier) :

```bash
cd dataset_tools
python drive_add_images.py --label hello_kitty --input ../mes_photos
```

Options : `--dry-run`, `--resize`, `--jpeg-quality`, `--keep-original-format`, `--background-color`, etc. Voir `python drive_add_images.py --help`.

## Tests

Depuis la racine du projet :

```bash
cd dataset_tools && pytest test_drive_add_images.py -v
```
