import argparse
import io
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# https://www.youtube.com/watch?v=HvKpzGadlB4

CLASSES = ["hello_kitty", "fake_hello_kitty", "other"]
LABELS = {"hello_kitty": 0, "fake_hello_kitty": 1, "other": 2}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SCOPES = ["https://www.googleapis.com/auth/drive"]


def moyenne(L):
    """Retourne la moyenne des éléments de la liste L(non vide)."""
    assert len(L) > 0, "La liste L est vide"
    S = 0
    for i in range(len(L)):
        S += L[i]
    return S / len(L)


def conversionNB(img_rgb):
    """Convertit RGB -> niveaux de gris (2D) avec boucles simples."""
    n = len(img_rgb)
    p = len(img_rgb[0])
    Z = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for j in range(p):
            Z[i][j] = moyenne(img_rgb[i][j])
    return Z


def contours(img_gray):
    """Contours simples: |vertical| + |horizontal|."""
    n = len(img_gray)
    p = len(img_gray[0])
    P = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(1, n - 1):
        for j in range(1, p - 1):
            P[i][j] = abs(img_gray[i + 1][j] - img_gray[i - 1][j]) + abs(img_gray[i][j + 1] - img_gray[i][j - 1])
            if P[i][j] > 255:
                P[i][j] = 255
    return P


def normaliser(img):
    """Divise tous les pixels par 255 (0..1)."""
    arr = np.asarray(img, dtype=np.float32)
    return arr / 255.0


def remplir_raw_depuis_sources(raw_dir, source_vrai, source_faux, source_other):
    """
    Copie les images locales dans datasets/raw (inspiré de la logique de filtrage image).
    """
    mapping = [
        ("hello_kitty", source_vrai),
        ("fake_hello_kitty", source_faux),
        ("other", source_other),
    ]
    for class_name, source in mapping:
        target = raw_dir / class_name
        target.mkdir(parents=True, exist_ok=True)
        if not source:
            continue
        source_path = Path(source)
        if not source_path.exists():
            print(f"[WARN] Source absente: {source_path}")
            continue
        for f in sorted(source_path.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                shutil.copy2(f, target / f.name)


def get_drive_service(credentials_path, token_path):
    creds = None
    credentials_path = Path(credentials_path)
    token_path = Path(token_path)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(f"credentials introuvable: {credentials_path}")
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def find_folder(service, parent_id, folder_name):
    q = (
        f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false and name='{folder_name}'"
    )
    resp = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
    files = resp.get("files", [])
    if not files:
        return None
    return files[0]["id"]


def list_drive_files(service, folder_id):
    q = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    resp = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
    return resp.get("files", [])


def download_drive_file(service, file_id, dst_path):
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dst_path.write_bytes(buffer.getvalue())


def remplir_raw_depuis_drive(raw_dir, drive_root_folder_name, credentials_path, token_path):
    """
    Télécharge depuis My Drive/{drive_root_folder_name}/{hello_kitty|fake_hello_kitty|other}
    vers datasets/raw.
    """
    service = get_drive_service(credentials_path, token_path)
    root_id = service.files().get(fileId="root", fields="id").execute()["id"]
    dataset_id = find_folder(service, root_id, drive_root_folder_name)
    if not dataset_id:
        raise FileNotFoundError(f"Dossier Drive introuvable: {drive_root_folder_name}")

    for class_name in CLASSES:
        class_id = find_folder(service, dataset_id, class_name)
        target = raw_dir / class_name
        target.mkdir(parents=True, exist_ok=True)
        if not class_id:
            print(f"[WARN] Dossier Drive absent pour classe: {class_name}")
            continue
        files = list_drive_files(service, class_id)
        downloaded = 0
        for f in files:
            name = f.get("name", "")
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            dst = target / name
            download_drive_file(service, f["id"], dst)
            downloaded += 1
        print(f"[Drive] {class_name}: {downloaded} fichier(s) téléchargé(s)")


def sauver_variant(base_dir, variante, normalisee, X, y):
    sous_dossier = "normalisee" if normalisee else "non_normalisee"
    out_dir = base_dir / variante / sous_dossier
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", np.asarray(X, dtype=np.float32))
    np.save(out_dir / "y.npy", np.asarray(y, dtype=np.int64))


def main():
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    dataset_tools_dir = project_root / "dataset_tools"

    parser = argparse.ArgumentParser(description="Construit datasets depuis Google Drive: raw + transformed.")
    parser.add_argument("--raw-dir", default=str(project_root / "datasets" / "raw"))
    parser.add_argument("--output-dir", default=str(project_root / "datasets" / "transformed"))
    parser.add_argument("--source-vrai", default="")
    parser.add_argument("--source-faux", default="")
    parser.add_argument("--source-other", default="")
    parser.add_argument("--skip-drive", action="store_true", help="Ne pas télécharger depuis Drive (mode local)")
    parser.add_argument("--drive-root-folder-name", default="dataset")
    parser.add_argument("--credentials", default=str(dataset_tools_dir / "credentials.json"))
    parser.add_argument("--token", default=str(dataset_tools_dir / "token.json"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionnel 1: copier depuis dossiers locaux.
    remplir_raw_depuis_sources(raw_dir, args.source_vrai, args.source_faux, args.source_other)

    # Par défaut: télécharger depuis Google Drive (dataset/hello_kitty|fake_hello_kitty|other).
    if not args.skip_drive:
        remplir_raw_depuis_drive(
            raw_dir,
            args.drive_root_folder_name,
            args.credentials,
            args.token,
        )

    X_rgb_raw = []
    X_rgb_norm = []
    X_nb_raw = []
    X_nb_norm = []
    X_contours_raw = []
    X_contours_norm = []
    y = []

    nb_par_classe = {c: 0 for c in CLASSES}
    rgb_shape_reference = None

    for classe in CLASSES:
        dossier_classe = raw_dir / classe
        if not dossier_classe.exists():
            continue
        for f in sorted(dossier_classe.iterdir()):
            if not (f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS):
                continue
            try:
                img = imageio.imread(f)
                pil = Image.fromarray(img).convert("RGB")
                img_rgb_np = np.asarray(pil, dtype=np.float32)
                if rgb_shape_reference is None:
                    rgb_shape_reference = img_rgb_np.shape
                if img_rgb_np.shape != rgb_shape_reference:
                    print(
                        f"[WARN] Taille différente ignorée ({f.name}): "
                        f"{img_rgb_np.shape} != {rgb_shape_reference}"
                    )
                    continue
                img_rgb = img_rgb_np.tolist()
            except Exception as e:
                print(f"[WARN] Image ignorée ({f.name}): {e}")
                continue

            img_nb = conversionNB(img_rgb)
            img_contours = contours(img_nb)

            X_rgb_raw.append(np.asarray(img_rgb, dtype=np.float32).reshape(-1))
            X_rgb_norm.append(normaliser(img_rgb).reshape(-1))
            X_nb_raw.append(np.asarray(img_nb, dtype=np.float32).reshape(-1))
            X_nb_norm.append(normaliser(img_nb).reshape(-1))
            X_contours_raw.append(np.asarray(img_contours, dtype=np.float32).reshape(-1))
            X_contours_norm.append(normaliser(img_contours).reshape(-1))
            y.append(LABELS[classe])
            nb_par_classe[classe] += 1

    sauver_variant(output_dir, "rgb", False, X_rgb_raw, y)
    sauver_variant(output_dir, "rgb", True, X_rgb_norm, y)
    sauver_variant(output_dir, "nb", False, X_nb_raw, y)
    sauver_variant(output_dir, "nb", True, X_nb_norm, y)
    sauver_variant(output_dir, "contours", False, X_contours_raw, y)
    sauver_variant(output_dir, "contours", True, X_contours_norm, y)

    print("Dataset construit.")
    print(f"Sortie: {output_dir}")
    print(f"Total images: {len(y)}")
    print(f"Par classe: {nb_par_classe}")
    if rgb_shape_reference is not None:
        print(f"Taille image conservée: {rgb_shape_reference}")
    if len(y) > 0:
        print(f"Taille vecteur RGB: {len(X_rgb_raw[0])}")
        print(f"Taille vecteur NB/contours: {len(X_nb_raw[0])}")


if __name__ == "__main__":
    main()

