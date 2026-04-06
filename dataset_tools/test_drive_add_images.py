"""
Tests pour drive_add_images.py.
Exécution : pytest test_drive_add_images.py -v
Mode mock : pas d'appel réel à l'API Google Drive.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import après éventuels mocks pour éviter de charger les credentials au démarrage
import drive_add_images as m


# ---------------------------------------------------------------------------
# Normalisation d'images
# ---------------------------------------------------------------------------


def test_parse_background_color_white_and_hex() -> None:
    assert m.parse_background_color("white") == (255, 255, 255)
    assert m.parse_background_color("#000000") == (0, 0, 0)
    assert m.parse_background_color("ff00aa") == (255, 0, 170)


def test_parse_background_color_invalid() -> None:
    with pytest.raises(ValueError, match="background-color"):
        m.parse_background_color("not_a_color")


def test_convert_transparency_to_rgb_on_white_background() -> None:
    from PIL import Image

    rgba = Image.new("RGBA", (2, 1), color=(255, 0, 0, 255))
    rgba.putpixel((0, 0), (255, 0, 0, 0))  # pixel totalement transparent

    out = m.convert_transparency_to_rgb(rgba, (255, 255, 255))
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((1, 0)) == (255, 0, 0)


def test_normalize_image_resize_128() -> None:
    from PIL import Image

    img = Image.new("RGB", (300, 100), color=(10, 20, 30))
    out = m.normalize_image(img, resize=128, no_resize=False, background_color="white")
    assert out.mode == "RGB"
    assert out.size == (128, 128)


def test_normalize_image_file_default_outputs_jpeg(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "in.png"
    Image.new("RGBA", (64, 64), color=(0, 255, 0, 128)).save(p)

    n = m.normalize_image_file(p)  # défaut = JPEG 128x128
    assert n.normalized_format == "JPEG"
    assert n.normalized_ext == ".jpg"
    assert n.normalized_size == (128, 128)
    assert n.normalized_bytes[:2] == b"\xff\xd8"  # signature JPEG


def test_normalize_image_file_keep_original_format_png(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "in.png"
    Image.new("RGBA", (64, 64), color=(0, 0, 255, 200)).save(p)

    n = m.normalize_image_file(p, keep_original_format=True)
    assert n.normalized_format == "PNG"
    assert n.normalized_ext == ".png"
    assert n.normalized_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # signature PNG


def test_same_normalized_image_gives_same_hashes(tmp_path: Path) -> None:
    """
    Deux images différentes (tailles différentes) mais couleur uniforme -> même résultat après normalisation -> mêmes hashes.
    """
    from PIL import Image

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGB", (1, 1), color=(123, 222, 10)).save(p1)
    Image.new("RGB", (50, 50), color=(123, 222, 10)).save(p2)

    n1 = m.normalize_image_file(p1, resize=128, jpeg_quality=90)
    n2 = m.normalize_image_file(p2, resize=128, jpeg_quality=90)

    md5_1 = m.compute_md5_from_bytes(n1.normalized_bytes)
    md5_2 = m.compute_md5_from_bytes(n2.normalized_bytes)
    assert md5_1 == md5_2

    ph1 = m.compute_phash_from_image(n1.normalized_pil)
    ph2 = m.compute_phash_from_image(n2.normalized_pil)
    assert ph1 == ph2


# ---------------------------------------------------------------------------
# Calcul des hashes
# ---------------------------------------------------------------------------


def test_compute_image_hashes_returns_tuple_of_strings(tmp_path: Path) -> None:
    """Vérifie que compute_image_hashes retourne (md5_hex, phash_hex)."""
    # Créer une petite image PNG (1x1 pixel)
    from PIL import Image
    img_path = tmp_path / "dot.png"
    img = Image.new("RGB", (1, 1), color=(128, 128, 128))
    img.save(img_path)

    md5_val, phash_val = m.compute_image_hashes(img_path)
    assert isinstance(md5_val, str)
    assert isinstance(phash_val, str)
    assert len(md5_val) == 32
    assert all(c in "0123456789abcdef" for c in md5_val)


def test_compute_image_hashes_file_not_found() -> None:
    """Fichier inexistant doit lever FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="introuvable"):
        m.compute_image_hashes("/nonexistent/image.png")


# ---------------------------------------------------------------------------
# Sélection des fichiers images dans un dossier
# ---------------------------------------------------------------------------


def test_collect_image_paths_single_file(tmp_path: Path) -> None:
    """Un fichier image unique retourne une liste d'un élément."""
    (tmp_path / "a.jpg").write_bytes(b"fake")
    paths = m.collect_image_paths(tmp_path / "a.jpg")
    assert len(paths) == 1
    assert paths[0].name == "a.jpg"


def test_collect_image_paths_unsupported_extension(tmp_path: Path) -> None:
    """Extension non supportée lève ValueError."""
    (tmp_path / "x.pdf").write_bytes(b"fake")
    with pytest.raises(ValueError, match="Extension non supportée"):
        m.collect_image_paths(tmp_path / "x.pdf")


def test_collect_image_paths_folder_filters_extensions(tmp_path: Path) -> None:
    """Un dossier : seuls jpg, jpeg, png, webp sont collectés."""
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "b.JPEG").write_bytes(b"2")
    (tmp_path / "c.png").write_bytes(b"3")
    (tmp_path / "d.webp").write_bytes(b"4")
    (tmp_path / "e.txt").write_bytes(b"5")
    (tmp_path / "f.gif").write_bytes(b"6")

    paths = m.collect_image_paths(tmp_path)
    names = {p.name for p in paths}
    assert names == {"a.jpg", "b.JPEG", "c.png", "d.webp"}
    assert "e.txt" not in names
    assert "f.gif" not in names


def test_collect_image_paths_nonexistent() -> None:
    """Chemin inexistant lève FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="introuvable"):
        m.collect_image_paths(tmp_path_that_does_not_exist := Path("/tmp/nonexistent_xyz_123"))


# ---------------------------------------------------------------------------
# Génération du prochain nom (parsing des noms existants)
# ---------------------------------------------------------------------------


def test_get_next_number_for_label_empty() -> None:
    """Aucun fichier existant -> prochain numéro = 1."""
    assert m.get_next_number_for_label([], "hello_kitty") == 1
    assert m.get_next_number_for_label([], "sanrio_other") == 1
    assert m.get_next_number_for_label([], "other") == 1


def test_get_next_number_for_label_from_existing() -> None:
    """Compteur basé sur les noms déjà présents."""
    names = ["hk_000001.jpg", "hk_000003.png", "hk_000002.webp"]
    assert m.get_next_number_for_label(names, "hello_kitty") == 4


def test_get_next_number_for_label_other_prefix() -> None:
    """other_ -> prefix 'other'."""
    assert m.get_next_number_for_label(["other_000010.jpg"], "other") == 11


def test_get_next_number_for_label_ignores_other_labels() -> None:
    """Les noms d'autres labels ne comptent pas."""
    names = ["hk_000001.jpg", "fhk_000100.jpg", "other_000050.jpg"]
    assert m.get_next_number_for_label(names, "hello_kitty") == 2
    assert m.get_next_number_for_label(names, "sanrio_other") == 101
    assert m.get_next_number_for_label(names, "other") == 51


# ---------------------------------------------------------------------------
# Déduplication : même md5/phash -> skip
# ---------------------------------------------------------------------------


def test_is_duplicate_by_md5() -> None:
    """Si un fichier existant a le même md5, c'est un doublon."""
    service = MagicMock()
    existing = [{"id": "x", "name": "hk_000001.jpg", "appProperties": {"md5": "abc123", "phash": "p1"}}]
    index = {}
    assert m.is_duplicate(service, "folder_id", existing, index, "abc123", "p2") is True


def test_is_duplicate_by_phash() -> None:
    """Si un fichier existant a le même phash, c'est un doublon."""
    service = MagicMock()
    existing = [{"id": "x", "name": "hk_000001.jpg", "appProperties": {"md5": "m1", "phash": "phash_same"}}]
    index = {}
    assert m.is_duplicate(service, "folder_id", existing, index, "other_md5", "phash_same") is True


def test_is_duplicate_no_match() -> None:
    """Pas de md5/phash commun -> pas doublon."""
    existing = [{"id": "x", "name": "hk_000001.jpg", "appProperties": {"md5": "a", "phash": "b"}}]
    index = {}
    assert m.is_duplicate(MagicMock(), "f", existing, index, "c", "d") is False


def test_is_duplicate_via_index() -> None:
    """Doublon détecté via index (by_md5 ou by_phash)."""
    existing = []
    index = {"by_md5": {"md5_in_index": "file_id"}, "by_phash": {}}
    assert m.is_duplicate(MagicMock(), "f", existing, index, "md5_in_index", "any_phash") is True
    index2 = {"by_md5": {}, "by_phash": {"phash_in_index": "file_id"}}
    assert m.is_duplicate(MagicMock(), "f", existing, index2, "any_md5", "phash_in_index") is True


# ---------------------------------------------------------------------------
# Dry-run : pas d'upload, pas d'appel Drive
# ---------------------------------------------------------------------------


@patch("drive_add_images.get_drive_service")
def test_run_dry_run_no_drive_calls(mock_get_service: MagicMock, tmp_path: Path) -> None:
    """En mode dry-run, get_drive_service n'est jamais appelé."""
    from PIL import Image
    img = tmp_path / "test.png"
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(img)

    m.run(label="hello_kitty", input_path=img, dry_run=True, log_path=tmp_path / "log.csv")
    mock_get_service.assert_not_called()


@patch("drive_add_images.get_drive_service")
def test_run_dry_run_with_folder(mock_get_service: MagicMock, tmp_path: Path) -> None:
    """Dry-run avec un dossier d'images : affiche les infos sans appeler Drive."""
    from PIL import Image
    Image.new("RGB", (1, 1), color=(1, 1, 1)).save(tmp_path / "a.png")
    Image.new("RGB", (1, 1), color=(2, 2, 2)).save(tmp_path / "b.png")

    m.run(label="other", input_path=tmp_path, dry_run=True, log_path=tmp_path / "log.csv")
    mock_get_service.assert_not_called()


# ---------------------------------------------------------------------------
# Mock complet : run avec service mocké (sans credentials)
# ---------------------------------------------------------------------------


@patch("drive_add_images.get_drive_service")
@patch("drive_add_images.get_or_create_dataset_folders")
@patch("drive_add_images.list_files_in_folder")
@patch("drive_add_images.fetch_index_from_drive")
@patch("drive_add_images.upload_image")
@patch("drive_add_images.append_log_row")
def test_run_upload_skips_duplicate(
    mock_append_log: MagicMock,
    mock_upload: MagicMock,
    mock_fetch_index: MagicMock,
    mock_list: MagicMock,
    mock_folders: MagicMock,
    mock_service: MagicMock,
    tmp_path: Path,
) -> None:
    """Si une image a le même md5 qu'un fichier existant, elle est skippée (pas d'upload)."""
    from PIL import Image
    img_path = tmp_path / "unique.png"
    Image.new("RGB", (3, 3), color=(1, 2, 3)).save(img_path)

    mock_service.return_value = MagicMock()
    mock_folders.return_value = {"hello_kitty": "fid_hello", "sanrio_other": "fid_fake", "other": "fid_other"}
    mock_fetch_index.return_value = {}

    # Normalisation mockée -> bytes déterministes
    from PIL import Image as PILImage
    normalized_bytes = b"normalized-bytes"
    normalized_md5 = m.compute_md5_from_bytes(normalized_bytes)
    normalized_phash = "phash_same"
    normalized_obj = m.NormalizedImage(
        original_path=img_path,
        normalized_bytes=normalized_bytes,
        normalized_pil=PILImage.new("RGB", (128, 128), color=(1, 2, 3)),
        normalized_format="JPEG",
        normalized_size=(128, 128),
        normalized_ext=".jpg",
        normalized_mime="image/jpeg",
    )

    mock_list.return_value = [
        {"id": "existing_id", "name": "hk_000001.jpg", "appProperties": {"md5": normalized_md5, "phash": normalized_phash}},
    ]

    with patch("drive_add_images.normalize_image_file", return_value=normalized_obj), patch(
        "drive_add_images.compute_phash_from_image", return_value=normalized_phash
    ):
        m.run(label="hello_kitty", input_path=img_path, dry_run=False, log_path=tmp_path / "log.csv")

    mock_upload.assert_not_called()
    mock_append_log.assert_called()
    # append_log_row(log_path, date, original_file, normalized_format, normalized_size, label, action, reason, drive_file_id)
    call_args = mock_append_log.call_args[0]
    assert call_args[6] == "skip"  # action
    assert "duplicate" in call_args[7].lower()  # reason


@patch("drive_add_images.get_drive_service")
@patch("drive_add_images.get_or_create_dataset_folders")
@patch("drive_add_images.list_files_in_folder")
@patch("drive_add_images.fetch_index_from_drive")
@patch("drive_add_images.upload_image")
@patch("drive_add_images.append_log_row")
def test_run_upload_new_file(
    mock_append_log: MagicMock,
    mock_upload: MagicMock,
    mock_fetch_index: MagicMock,
    mock_list: MagicMock,
    mock_folders: MagicMock,
    mock_service: MagicMock,
    tmp_path: Path,
) -> None:
    """Une image nouvelle est uploadée avec le bon nom (hk_000001.jpg par défaut)."""
    from PIL import Image
    img_path = tmp_path / "new.png"
    Image.new("RGB", (5, 5), color=(10, 20, 30)).save(img_path)

    mock_service.return_value = MagicMock()
    mock_folders.return_value = {"hello_kitty": "fid_hello", "sanrio_other": "fid_fake", "other": "fid_other"}
    mock_list.return_value = []
    mock_fetch_index.return_value = {}
    mock_upload.return_value = "new_file_id_123"

    # Normalisation mockée pour contrôler l'extension et les bytes envoyés
    normalized_bytes = b"normalized"
    from PIL import Image as PILImage
    normalized_obj = m.NormalizedImage(
        original_path=img_path,
        normalized_bytes=normalized_bytes,
        normalized_pil=PILImage.new("RGB", (128, 128), color=(10, 20, 30)),
        normalized_format="JPEG",
        normalized_size=(128, 128),
        normalized_ext=".jpg",
        normalized_mime="image/jpeg",
    )
    with patch("drive_add_images.normalize_image_file", return_value=normalized_obj), patch(
        "drive_add_images.compute_phash_from_image", return_value="ph"
    ):
        m.run(label="hello_kitty", input_path=img_path, dry_run=False, log_path=tmp_path / "log.csv")

    mock_upload.assert_called_once()
    call_kw = mock_upload.call_args
    assert call_kw[0][2] == "hk_000001.jpg"  # drive_file_name
    assert call_kw[0][3] == normalized_bytes  # normalized_bytes
    assert call_kw[0][4] == "image/jpeg"  # normalized_mime
    mock_append_log.assert_called()
    # append_log_row(log_path, date, original_file, normalized_format, normalized_size, label, action, reason, drive_file_id)
    args = mock_append_log.call_args[0]
    assert args[6] == "uploaded"  # action
    assert args[8] == "new_file_id_123"  # drive_file_id


# ---------------------------------------------------------------------------
# Label invalide
# ---------------------------------------------------------------------------


def test_run_invalid_label() -> None:
    """Label non autorisé lève ValueError."""
    with pytest.raises(ValueError, match="Label invalide"):
        m.run(label="invalid_label", input_path=Path("/tmp"), dry_run=True)


# ---------------------------------------------------------------------------
# Log CSV
# ---------------------------------------------------------------------------


def test_append_log_row_creates_file_with_headers(tmp_path: Path) -> None:
    """Premier appel crée le fichier avec en-têtes."""
    log_file = tmp_path / "log.csv"
    m.append_log_row(log_file, "2025-01-01T12:00:00", "img.png", "JPEG", (128, 128), "hello_kitty", "uploaded", "", "fid1")
    content = log_file.read_text(encoding="utf-8")
    assert "date,original_file,normalized_format,normalized_size,label,action,reason,drive_file_id" in content
    assert "img.png" in content
    assert "uploaded" in content
