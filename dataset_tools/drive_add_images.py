#!/usr/bin/env python3
"""
Script pour ajouter des images dans Google Drive (Mon Drive/dataset/{hello_kitty|sanrio_other|other}).
Gestion des doublons (md5 + phash), renommage automatique, log CSV.
Authentification OAuth 2.0 (credentials.json + token.json).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from PIL import Image, ImageOps, UnidentifiedImageError
import imagehash
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
INDEX_FILENAME = "dataset_index.json"
LOG_FILENAME = "drive_add_log.csv"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_RESIZE = 128
DEFAULT_JPEG_QUALITY = 90

LABEL_TO_PREFIX = {
    "hello_kitty": "hk",
    "sanrio_other": "sanrio",
    "other": "other",
}


# ---------------------------------------------------------------------------
# Normalisation, hashes et fichiers images
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedImage:
    original_path: Path
    normalized_bytes: bytes
    normalized_pil: Image.Image
    normalized_format: str  # "JPEG" | "PNG" | "WEBP"
    normalized_size: tuple[int, int]
    normalized_ext: str  # ".jpg" | ".png" | ".webp"
    normalized_mime: str  # "image/jpeg" | ...


def parse_background_color(value: str) -> tuple[int, int, int]:
    """
    Supporte: 'white', 'black', '#RRGGBB'
    """
    v = (value or "").strip().lower()
    if v in ("white", "blanc"):
        return (255, 255, 255)
    if v in ("black", "noir"):
        return (0, 0, 0)
    if re.fullmatch(r"#?[0-9a-f]{6}", v):
        vv = v[1:] if v.startswith("#") else v
        return (int(vv[0:2], 16), int(vv[2:4], 16), int(vv[4:6], 16))
    raise ValueError("background-color invalide. Utilisez white|black|#RRGGBB.")


def convert_transparency_to_rgb(img: Image.Image, background_rgb: tuple[int, int, int]) -> Image.Image:
    """
    Convertit une image potentiellement transparente en RGB sur fond uni.
    """
    # Palette (P) -> RGBA si transparence
    if img.mode == "P":
        img = img.convert("RGBA")

    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, background_rgb)
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg

    # Pas de transparence : conversion simple
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def normalize_image(
    img: Image.Image,
    *,
    resize: int = DEFAULT_RESIZE,
    no_resize: bool = False,
    background_color: str = "white",
) -> Image.Image:
    """
    Pipeline ML: EXIF transpose, transparence -> fond uni, RGB, resize carré (optionnel).
    """
    img = ImageOps.exif_transpose(img)
    bg_rgb = parse_background_color(background_color)
    img_rgb = convert_transparency_to_rgb(img, bg_rgb)

    if not no_resize:
        if resize <= 0:
            raise ValueError("--resize doit être un entier > 0")
        img_rgb = ImageOps.fit(img_rgb, (resize, resize), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    return img_rgb


def _decide_output_format_and_ext(original_path: Path, keep_original_format: bool) -> tuple[str, str, str]:
    """
    Retourne (format, ext, mime).
    - Par défaut: JPEG + .jpg
    - Sinon: conserve PNG/WEBP/JPEG si possible, sinon JPEG
    """
    if not keep_original_format:
        return ("JPEG", ".jpg", "image/jpeg")

    ext = original_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return ("JPEG", ".jpg", "image/jpeg")
    if ext == ".png":
        return ("PNG", ".png", "image/png")
    if ext == ".webp":
        return ("WEBP", ".webp", "image/webp")
    return ("JPEG", ".jpg", "image/jpeg")


def save_normalized_to_bytes(
    img_rgb: Image.Image,
    *,
    output_format: str,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """
    Sérialise l'image normalisée en bytes selon output_format.
    Pour JPEG, on fixe des options pour éviter des variations inutiles.
    """
    buf = io.BytesIO()
    fmt = output_format.upper()
    if fmt == "JPEG":
        q = int(jpeg_quality)
        if q < 1 or q > 95:
            raise ValueError("--jpeg-quality doit être entre 1 et 95 (Pillow)")
        img_rgb.save(
            buf,
            format="JPEG",
            quality=q,
            optimize=False,
            progressive=False,
            subsampling=0,
        )
    elif fmt == "PNG":
        img_rgb.save(buf, format="PNG", optimize=True)
    elif fmt == "WEBP":
        # Qualité WebP alignée sur jpeg_quality
        q = int(jpeg_quality)
        if q < 0 or q > 100:
            raise ValueError("--jpeg-quality doit être entre 0 et 100 pour WEBP")
        img_rgb.save(buf, format="WEBP", quality=q, method=6)
    else:
        raise ValueError(f"Format de sortie non supporté: {output_format}")

    return buf.getvalue()


def compute_md5_from_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def compute_phash_from_image(img_rgb: Image.Image) -> str:
    return str(imagehash.phash(img_rgb))


def normalize_image_file(
    file_path: str | Path,
    *,
    resize: int = DEFAULT_RESIZE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    no_resize: bool = False,
    keep_original_format: bool = False,
    background_color: str = "white",
) -> NormalizedImage:
    """
    Ouvre un fichier image et produit une version normalisée (bytes + PIL).
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Fichier introuvable: {p}")

    output_format, output_ext, output_mime = _decide_output_format_and_ext(p, keep_original_format)

    with Image.open(p) as img:
        img.load()
        img_rgb = normalize_image(img, resize=resize, no_resize=no_resize, background_color=background_color)

    data = save_normalized_to_bytes(img_rgb, output_format=output_format, jpeg_quality=jpeg_quality)
    return NormalizedImage(
        original_path=p,
        normalized_bytes=data,
        normalized_pil=img_rgb,
        normalized_format=output_format,
        normalized_size=img_rgb.size,
        normalized_ext=output_ext,
        normalized_mime=output_mime,
    )


def compute_image_hashes(file_path: str | Path) -> tuple[str, str]:
    """
    Calcule md5 + phash sur l'image *normalisée* (pipeline par défaut).

    :param file_path: Chemin vers le fichier image.
    :return: (md5_hex, phash_hex).
    :raises OSError: Si le fichier ne peut pas être lu.
    """
    n = normalize_image_file(file_path)
    md5_hash = compute_md5_from_bytes(n.normalized_bytes)
    phash_hex = compute_phash_from_image(n.normalized_pil)
    return md5_hash, phash_hex


def collect_image_paths(input_path: str | Path) -> list[Path]:
    """
    Retourne la liste des chemins vers les fichiers images à traiter.
    Si input_path est un fichier : retourne [input_path] si extension supportée.
    Si c'est un dossier : retourne tous les fichiers avec extension supportée (récursif optionnel : non par défaut).

    :param input_path: Fichier ou dossier.
    :return: Liste de Path.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Chemin introuvable: {path}")

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        raise ValueError(f"Extension non supportée: {path.suffix}. Utilisez: {SUPPORTED_EXTENSIONS}")

    if path.is_dir():
        collected = []
        for f in path.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                collected.append(f)
        return sorted(collected)

    return []


def get_next_number_for_label(existing_names: list[str], label: str) -> int:
    """
    À partir des noms de fichiers déjà présents dans le dossier Drive pour ce label,
    détermine le prochain numéro à utiliser (format: prefix_NNNNNN.ext).

    :param existing_names: Noms de fichiers (ex: ["hk_000001.jpg", "hk_000003.png"]).
    :param label: Clé du label (hello_kitty, sanrio_other, other).
    :return: Prochain numéro (1-based, ex: 4).
    """
    prefix = LABEL_TO_PREFIX.get(label, "other")
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.[a-zA-Z]+$", re.IGNORECASE)
    numbers = []
    for name in existing_names:
        m = pattern.match(name)
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers, default=0) + 1


# ---------------------------------------------------------------------------
# Google Drive : authentification et service
# ---------------------------------------------------------------------------


def get_drive_service(credentials_path: str | Path = CREDENTIALS_FILE, token_path: str | Path = TOKEN_FILE):
    """
    Authentification OAuth 2.0 et construction du service Drive.
    Utilise credentials.json et token.json (créé après premier flux).

    :return: Resource (service Drive API).
    """
    creds = None
    token_path = Path(token_path)
    credentials_path = Path(credentials_path)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Fichier {credentials_path} introuvable. "
                    "Téléchargez les credentials OAuth depuis Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Drive : dossiers et fichiers
# ---------------------------------------------------------------------------


def find_or_create_folder(service: Any, parent_id: str, folder_name: str) -> str:
    """Trouve un dossier par nom sous parent_id, ou le crée. Retourne l'id du dossier."""
    q = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    resp = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
    for f in resp.get("files", []):
        if f["name"] == folder_name:
            return f["id"]

    meta = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def get_or_create_dataset_folders(service: Any, drive_root_folder_name: str = "dataset") -> dict[str, str]:
    """
    Retourne les ids des dossiers dataset/hello_kitty, dataset/sanrio_other, dataset/other.
    Crée l'arborescence sous "My Drive" si nécessaire.
    """
    # Racine "My Drive" = root
    root_resp = service.files().get(fileId="root", fields="id").execute()
    root_id = root_resp["id"]

    dataset_id = find_or_create_folder(service, root_id, drive_root_folder_name)
    return {
        "hello_kitty": find_or_create_folder(service, dataset_id, "hello_kitty"),
        "sanrio_other": find_or_create_folder(service, dataset_id, "sanrio_other"),
        "other": find_or_create_folder(service, dataset_id, "other"),
    }


def list_files_in_folder(service: Any, folder_id: str) -> list[dict]:
    """Liste les fichiers dans un dossier (non supprimés). Retourne id, name, appProperties."""
    q = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    resp = service.files().list(q=q, spaces="drive", fields="files(id, name, appProperties)").execute()
    return resp.get("files", [])


def fetch_index_from_drive(service: Any, folder_ids: dict[str, str]) -> dict:
    """Récupère dataset_index.json depuis le premier dossier où il existe (ex: hello_kitty)."""
    for label, fid in folder_ids.items():
        q = f"name='{INDEX_FILENAME}' and '{fid}' in parents and trashed=false"
        resp = service.files().list(q=q, spaces="drive", fields="files(id)").execute()
        files = resp.get("files", [])
        if files:
            from io import BytesIO
            from googleapiclient.http import MediaIoBaseDownload
            fh = service.files().get_media(fileId=files[0]["id"])
            buf = BytesIO()
            downloader = MediaIoBaseDownload(buf, fh)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buf.seek(0)
            return json.load(buf)
    return {"by_md5": {}, "by_phash": {}, "files": []}


def is_duplicate(
    service: Any,
    folder_id: str,
    existing_files: list[dict],
    index: dict,
    md5: str,
    phash: str,
) -> bool:
    """
    Vérifie si (md5 ou phash) existe déjà dans le dossier (appProperties) ou dans l'index.
    existing_files = list_files_in_folder(...).
    """
    for f in existing_files:
        props = f.get("appProperties") or {}
        if props.get("md5") == md5 or props.get("phash") == phash:
            return True
    if index:
        if md5 in index.get("by_md5", {}):
            return True
        if phash in index.get("by_phash", {}):
            return True
    return False


def upload_image(
    service: Any,
    folder_id: str,
    drive_file_name: str,
    normalized_bytes: bytes,
    normalized_mime: str,
    md5: str,
    phash: str,
    original_file: str,
    normalized_format: str,
    normalized_size: tuple[int, int],
    added_by: str = "",
    source: str = "",
) -> str:
    """
    Upload un fichier dans le dossier Drive avec appProperties (md5, phash, added_at, added_by, source, normalisation).
    Retourne l'id du fichier créé.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    body = {
        "name": drive_file_name,
        "parents": [folder_id],
        "appProperties": {
            "md5": md5,
            "phash": phash,
            "added_at": now_iso,
            "added_by": added_by or "",
            "source": source or "",
            "original_file": original_file,
            "normalized_format": normalized_format,
            "normalized_size": f"{normalized_size[0]}x{normalized_size[1]}",
        },
    }
    media = MediaIoBaseUpload(io.BytesIO(normalized_bytes), mimetype=normalized_mime, resumable=True)
    created = service.files().create(body=body, media_body=media, fields="id").execute()
    return created["id"]


# ---------------------------------------------------------------------------
# Log CSV
# ---------------------------------------------------------------------------

LOG_HEADERS = [
    "date",
    "original_file",
    "normalized_format",
    "normalized_size",
    "label",
    "action",
    "reason",
    "drive_file_id",
]

def _ensure_log_file(log_path: str | Path) -> Path:
    """
    Assure que le fichier de log a les bons en-têtes.
    Si un ancien log existe avec d'autres en-têtes, on écrit dans un nouveau fichier *_v2.csv.
    """
    p = Path(log_path)
    expected = ",".join(LOG_HEADERS)
    if not p.exists() or p.stat().st_size == 0:
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADERS)
        return p

    try:
        first = p.read_text(encoding="utf-8").splitlines()[0].strip()
    except Exception:
        first = ""

    if first == expected:
        return p

    # Ancienne version ou fichier custom : bascule sur un nouveau log
    alt = p.with_name(f"{p.stem}_v2{p.suffix}")
    if not alt.exists() or alt.stat().st_size == 0:
        with open(alt, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADERS)
    return alt


def append_log_row(
    log_path: str | Path,
    date: str,
    original_file: str,
    normalized_format: str,
    normalized_size: tuple[int, int],
    label: str,
    action: str,
    reason: str,
    drive_file_id: str = "",
) -> None:
    """Ajoute une ligne au fichier log CSV (création du fichier avec en-têtes si nécessaire)."""
    path = _ensure_log_file(log_path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                date,
                original_file,
                normalized_format,
                f"{normalized_size[0]}x{normalized_size[1]}",
                label,
                action,
                reason,
                drive_file_id,
            ]
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _is_supported_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


def get_dataset_counts(service: Any, folder_ids: dict[str, str]) -> dict[str, int]:
    """
    Compte le nombre d'images (extensions supportées) dans chaque dossier label.
    """
    counts: dict[str, int] = {}
    for label, folder_id in folder_ids.items():
        try:
            files = list_files_in_folder(service, folder_id)
            counts[label] = sum(1 for f in files if _is_supported_image_name(f.get("name", "")))
        except Exception:
            # On n'interrompt pas le flux d'upload pour un simple problème de comptage.
            counts[label] = -1
    return counts


def run(
    label: str,
    input_path: str | Path,
    drive_root_folder_name: str = "dataset",
    dry_run: bool = False,
    added_by: str = "",
    source: str = "",
    resize: int = DEFAULT_RESIZE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    no_resize: bool = False,
    keep_original_format: bool = False,
    background_color: str = "white",
    credentials_path: str | Path = CREDENTIALS_FILE,
    token_path: str | Path = TOKEN_FILE,
    log_path: str | Path = LOG_FILENAME,
) -> None:
    """
    Point d'entrée principal : collecte les images, vérifie doublons, upload (ou dry-run).
    """
    if label not in LABEL_TO_PREFIX:
        raise ValueError(f"Label invalide: {label}. Attendu: {list(LABEL_TO_PREFIX)}")

    paths = collect_image_paths(input_path)
    if not paths:
        print("Aucune image à traiter.")
        return

    if dry_run:
        print(f"[DRY-RUN] {len(paths)} image(s) à traiter pour le label '{label}'.")
        for p in paths:
            try:
                n = normalize_image_file(
                    p,
                    resize=resize,
                    jpeg_quality=jpeg_quality,
                    no_resize=no_resize,
                    keep_original_format=keep_original_format,
                    background_color=background_color,
                )
                md5 = compute_md5_from_bytes(n.normalized_bytes)
                phash = compute_phash_from_image(n.normalized_pil)
                print(
                    f"  - {p.name} -> normalized={n.normalized_format} {n.normalized_size[0]}x{n.normalized_size[1]} "
                    f"md5={md5[:12]}... phash={phash}"
                )
            except Exception as e:
                print(f"  - {p.name} -> Erreur: {e}")
        return

    service = get_drive_service(credentials_path, token_path)
    folders = get_or_create_dataset_folders(service, drive_root_folder_name)
    folder_id = folders[label]
    existing_files = list_files_in_folder(service, folder_id)
    existing_names = [f["name"] for f in existing_files]
    index = fetch_index_from_drive(service, folders)

    next_num = get_next_number_for_label(existing_names, label)
    prefix = LABEL_TO_PREFIX[label]
    now_iso = datetime.now(timezone.utc).isoformat()
    stats = {"uploaded": 0, "skip": 0, "error": 0}
    resolved_log_path = _ensure_log_file(log_path)

    for path in tqdm(paths, desc="Upload"):
        try:
            n = normalize_image_file(
                path,
                resize=resize,
                jpeg_quality=jpeg_quality,
                no_resize=no_resize,
                keep_original_format=keep_original_format,
                background_color=background_color,
            )
            md5 = compute_md5_from_bytes(n.normalized_bytes)
            phash = compute_phash_from_image(n.normalized_pil)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            append_log_row(resolved_log_path, now_iso, path.name, "", (0, 0), label, "error", str(e), "")
            tqdm.write(f"Erreur image {path.name}: {e}")
            stats["error"] += 1
            continue
        except Exception as e:
            append_log_row(resolved_log_path, now_iso, path.name, "", (0, 0), label, "error", str(e), "")
            tqdm.write(f"Erreur {path.name}: {e}")
            stats["error"] += 1
            continue

        if is_duplicate(service, folder_id, existing_files, index, md5, phash):
            append_log_row(
                resolved_log_path,
                now_iso,
                path.name,
                n.normalized_format,
                n.normalized_size,
                label,
                "skip",
                "duplicate",
                "",
            )
            tqdm.write(f"Doublon ignoré: {path.name} (après normalisation)")
            stats["skip"] += 1
            continue

        drive_name = f"{prefix}_{next_num:06d}{n.normalized_ext}"
        file_id = upload_image(
            service,
            folder_id,
            drive_name,
            n.normalized_bytes,
            n.normalized_mime,
            md5,
            phash,
            original_file=path.name,
            normalized_format=n.normalized_format,
            normalized_size=n.normalized_size,
            added_by=added_by,
            source=source,
        )
        append_log_row(
            resolved_log_path,
            now_iso,
            path.name,
            n.normalized_format,
            n.normalized_size,
            label,
            "uploaded",
            "",
            file_id,
        )
        existing_files.append({"id": file_id, "name": drive_name, "appProperties": {"md5": md5, "phash": phash}})
        existing_names.append(drive_name)
        index.setdefault("by_md5", {})[md5] = file_id
        index.setdefault("by_phash", {})[phash] = file_id
        next_num += 1
        stats["uploaded"] += 1

    if Path(resolved_log_path) != Path(log_path):
        print(f"Ancien log détecté, nouveau log écrit dans {resolved_log_path}.")
    else:
        print(f"Log écrit dans {resolved_log_path}.")
    print(f"Session: uploaded={stats['uploaded']} skip={stats['skip']} error={stats['error']}")

    counts = get_dataset_counts(service, folders)
    print("Récap dataset (Drive) :")
    for k in ("hello_kitty", "sanrio_other", "other"):
        v = counts.get(k, 0)
        if v == -1:
            print(f"  - {k}: (indisponible)")
        else:
            print(f"  - {k}: {v} image(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ajoute des images dans Google Drive (dataset/hello_kitty|sanrio_other|other)."
    )
    parser.add_argument("--label", required=True, choices=list(LABEL_TO_PREFIX), help="Classe: hello_kitty | sanrio_other | other")
    parser.add_argument("--input", required=True, help="Fichier image ou dossier contenant des images")
    parser.add_argument("--drive-root-folder-name", default="dataset", help="Nom du dossier racine sous My Drive (défaut: dataset)")
    parser.add_argument("--dry-run", action="store_true", help="Afficher les actions sans upload")
    parser.add_argument("--added-by", default="", help="Identifiant de la personne qui ajoute (stocké dans appProperties)")
    parser.add_argument("--source", default="", help="Source optionnelle (url/texte)")
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE, help=f"Taille carrée de sortie (défaut: {DEFAULT_RESIZE})")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help=f"Qualité JPEG (défaut: {DEFAULT_JPEG_QUALITY})")
    parser.add_argument("--no-resize", action="store_true", help="Désactive le redimensionnement (garde la taille après conversion RGB)")
    parser.add_argument("--keep-original-format", action="store_true", help="Conserve le format source (PNG/WebP/JPEG) au lieu de forcer le JPEG")
    parser.add_argument("--background-color", default="white", help="Couleur de fond pour transparence: white|black|#RRGGBB (défaut: white)")
    parser.add_argument("--credentials", default=CREDENTIALS_FILE, help=f"Fichier credentials OAuth (défaut: {CREDENTIALS_FILE})")
    parser.add_argument("--token", default=TOKEN_FILE, help=f"Fichier token OAuth (défaut: {TOKEN_FILE})")
    parser.add_argument("--log", default=LOG_FILENAME, help=f"Fichier log CSV (défaut: {LOG_FILENAME})")
    args = parser.parse_args()

    run(
        label=args.label,
        input_path=args.input,
        drive_root_folder_name=args.drive_root_folder_name,
        dry_run=args.dry_run,
        added_by=args.added_by,
        source=args.source,
        resize=args.resize,
        jpeg_quality=args.jpeg_quality,
        no_resize=args.no_resize,
        keep_original_format=args.keep_original_format,
        background_color=args.background_color,
        credentials_path=args.credentials,
        token_path=args.token,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
