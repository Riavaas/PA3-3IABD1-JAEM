from __future__ import annotations

import ctypes
import math
import os
import platform
import shutil
import threading
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_DATA_DIR = PROJECT_ROOT / "datasets" / "transformed" / "nb" / "normalisee"
RGB_DATA_DIR = PROJECT_ROOT / "datasets" / "transformed" / "rgb" / "normalisee"
CACHE_DIR = PROJECT_ROOT / "models" / "cache_gradio"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ces anciens caches contiennent des paramètres incompatibles avec la configuration actuelle.
LEGACY_CACHE_FILES = (
    CACHE_DIR / "linear_nb_normalisee.npz",
    CACHE_DIR / "rbf_nb_normalisee.npz",
)
for legacy_cache in LEGACY_CACHE_FILES:
    legacy_cache.unlink(missing_ok=True)

LABELS = {
    0: "Route sèche",
    1: "Route mouillée",
    2: "Route enneigée",
}


def resolve_library_path(directory: Path, stem: str) -> Path:
    system = platform.system()
    if system == "Windows":
        # MinGW peut produire un binaire PE Windows avec le suffixe .so.
        suffixes = [".dll", ".so"]
    elif system == "Darwin":
        suffixes = [".dylib", ".so"]
    else:
        suffixes = [".so"]

    candidates = [directory / f"{stem}{suffix}" for suffix in suffixes]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODEL_CONFIG = {
    "Modèle linéaire": {
        "library": resolve_library_path(
            PROJECT_ROOT / "models" / "lineaire", "linear_model_lib"
        ),
        "cache": CACHE_DIR / "linear_nb_normalisee_pocket_e100_lr001.npz",
    },
    "MLP": {
        "library": resolve_library_path(PROJECT_ROOT / "models" / "mlp", "mlp_lib"),
        "cache": CACHE_DIR / "mlp_nb_normalisee.npz",
    },
    "RBF": {
        "library": resolve_library_path(PROJECT_ROOT / "models" / "rbf", "rbf_lib"),
        "cache": CACHE_DIR / "rbf_rgb_normalisee_rosenblatt_k400_g00001_e30_lr001.npz",
    },
}

_models: dict[str, dict[str, object]] = {}
_models_lock = threading.Lock()
_dll_directory_handles: list[object] = []


def read_x(path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
    with path.open("rb") as file:
        n = int(np.fromfile(file, dtype=np.int32, count=1)[0])
        d = int(np.fromfile(file, dtype=np.int32, count=1)[0])
        values = np.fromfile(file, dtype=np.float32, count=n * d)
    if values.size != n * d:
        raise RuntimeError(f"Fichier X incomplet : {path}")
    return np.ascontiguousarray(values.reshape(n, d).astype(dtype, copy=False))


def read_y(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        n = int(np.fromfile(file, dtype=np.int32, count=1)[0])
        values = np.fromfile(file, dtype=np.int32, count=n)
    if values.size != n:
        raise RuntimeError(f"Fichier y incomplet : {path}")
    return np.ascontiguousarray(values)


def require_file(path: Path, message: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{message}\nFichier attendu : {path}")


def load_library(path: Path, model_name: str) -> ctypes.CDLL:
    require_file(
        path,
        f"La bibliothèque du modèle {model_name} n'est pas compilée pour {platform.system()}. "
        "Exécutez : python site_gradio/build_libs.py",
    )
    try:
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            # Le dossier de la bibliothèque est toujours ajouté.
            _dll_directory_handles.append(os.add_dll_directory(str(path.parent)))

            # Les bibliothèques C++ produites par MinGW peuvent dépendre de
            # libgcc_s_seh-1.dll et libstdc++-6.dll. Python 3.8+ ne recherche
            # pas automatiquement ces DLL dans tous les dossiers du PATH.
            compiler = shutil.which("g++")
            if compiler:
                compiler_directory = Path(compiler).resolve().parent
                _dll_directory_handles.append(
                    os.add_dll_directory(str(compiler_directory))
                )

        return ctypes.CDLL(str(path))
    except OSError as exc:
        raise RuntimeError(
            f"Impossible de charger {path.name}. La bibliothèque existe, mais une dépendance native "
            "peut manquer. Recompilez-la avec : python site_gradio/build_libs.py\n"
            f"Détail système : {exc}"
        ) from exc


def load_training_data(
    data_dir: Path,
    representation_name: str,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    x_path = data_dir / "X_train.f32bin"
    y_path = data_dir / "y_train.i32bin"
    require_file(x_path, f"Le jeu d'entraînement {representation_name} est absent.")
    require_file(y_path, f"Les étiquettes d'entraînement {representation_name} sont absentes.")
    return read_x(x_path, dtype), read_y(y_path)


def load_linear() -> dict[str, object]:
    cfg = MODEL_CONFIG["Modèle linéaire"]
    cache_path = Path(cfg["cache"])
    if cache_path.exists():
        saved = np.load(cache_path)
        return {"W": saved["W"], "b": saved["b"], "d": int(saved["d"]), "from_cache": True}

    lib = load_library(Path(cfg["library"]), "linéaire")
    pf = ctypes.POINTER(ctypes.c_float)
    pi = ctypes.POINTER(ctypes.c_int)
    lib.fit.argtypes = [pf, pi, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, pf, pf]
    lib.fit.restype = None

    X, y = load_training_data(NB_DATA_DIR, "NB normalisé", np.float32)
    d = X.shape[1]
    W = np.zeros(3 * d, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    lib.fit(
        X.ctypes.data_as(pf), y.ctypes.data_as(pi), X.shape[0], d,
        100, ctypes.c_float(0.01), W.ctypes.data_as(pf), b.ctypes.data_as(pf),
    )
    np.savez_compressed(cache_path, W=W, b=b, d=d)
    return {"W": W, "b": b, "d": d, "from_cache": False}


def load_mlp() -> dict[str, object]:
    cfg = MODEL_CONFIG["MLP"]
    cache_path = Path(cfg["cache"])
    if cache_path.exists():
        saved = np.load(cache_path)
        return {
            "W1": saved["W1"], "b1": saved["b1"],
            "W2": saved["W2"], "b2": saved["b2"],
            "d": int(saved["d"]), "H": int(saved["H"]), "from_cache": True,
        }

    lib = load_library(Path(cfg["library"]), "MLP")
    pf = ctypes.POINTER(ctypes.c_float)
    pd = ctypes.POINTER(ctypes.c_double)
    pi = ctypes.POINTER(ctypes.c_int)
    lib.entrainer.argtypes = [
        pf, pi, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_uint, pd, pd, pd, pd,
    ]
    lib.entrainer.restype = None

    X, y = load_training_data(NB_DATA_DIR, "NB normalisé", np.float32)
    d, H = X.shape[1], 32
    W1 = np.zeros(H * d, dtype=np.float64)
    b1 = np.zeros(H, dtype=np.float64)
    W2 = np.zeros(3 * H, dtype=np.float64)
    b2 = np.zeros(3, dtype=np.float64)
    lib.entrainer(
        X.ctypes.data_as(pf), y.ctypes.data_as(pi), X.shape[0], d, H,
        30, ctypes.c_double(0.001), ctypes.c_uint(67),
        W1.ctypes.data_as(pd), b1.ctypes.data_as(pd),
        W2.ctypes.data_as(pd), b2.ctypes.data_as(pd),
    )
    np.savez_compressed(cache_path, W1=W1, b1=b1, W2=W2, b2=b2, d=d, H=H)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "d": d, "H": H, "from_cache": False}


def load_rbf() -> dict[str, object]:
    cfg = MODEL_CONFIG["RBF"]
    cache_path = Path(cfg["cache"])
    if cache_path.exists():
        saved = np.load(cache_path)
        return {
            "centres": saved["centres"], "W": saved["W"], "b": saved["b"],
            "d": int(saved["d"]), "nb_centres": int(saved["nb_centres"]),
            "gamma": float(saved["gamma"]), "from_cache": True,
        }

    lib = load_library(Path(cfg["library"]), "RBF")
    pd = ctypes.POINTER(ctypes.c_double)
    pi = ctypes.POINTER(ctypes.c_int)
    lib.entrainer_rosenblatt.argtypes = [
        pd, pi, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_uint, ctypes.c_int, ctypes.c_double,
        pd, pd, pd,
    ]
    lib.entrainer_rosenblatt.restype = None

    X, y = load_training_data(RGB_DATA_DIR, "RGB normalisé", np.float64)
    d = X.shape[1]
    nb_centres, gamma, epochs, lr = min(400, X.shape[0]), 0.0001, 30, 0.01
    centres = np.zeros(nb_centres * d, dtype=np.float64)
    W = np.zeros(3 * nb_centres, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    lib.entrainer_rosenblatt(
        X.ctypes.data_as(pd), y.ctypes.data_as(pi), X.shape[0], d,
        nb_centres, ctypes.c_double(gamma), ctypes.c_uint(42),
        epochs, ctypes.c_double(lr),
        centres.ctypes.data_as(pd), W.ctypes.data_as(pd), b.ctypes.data_as(pd),
    )
    np.savez_compressed(
        cache_path, centres=centres, W=W, b=b, d=d,
        nb_centres=nb_centres, gamma=gamma, epochs=epochs, lr=lr,
    )
    return {
        "centres": centres, "W": W, "b": b, "d": d,
        "nb_centres": nb_centres, "gamma": gamma, "from_cache": False,
    }


def get_model(model_name: str) -> tuple[dict[str, object], bool]:
    with _models_lock:
        if model_name in _models:
            return _models[model_name], True
        loaders = {
            "Modèle linéaire": load_linear,
            "MLP": load_mlp,
            "RBF": load_rbf,
        }
        model = loaders[model_name]()
        _models[model_name] = model
        return model, False


def preprocess_image(
    image: Image.Image,
    d: int,
    dtype: np.dtype,
    representation: str = "nb",
) -> np.ndarray:
    if representation == "rgb":
        if d % 3 != 0:
            raise RuntimeError(f"Dimension RGB invalide : d={d}")
        side = int(math.isqrt(d // 3))
        if side * side * 3 != d:
            raise RuntimeError(f"Dimension RGB incompatible avec une image carrée : d={d}")
        rgb = image.convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
        values = np.asarray(rgb, dtype=np.float64) / 255.0
    else:
        side = int(math.isqrt(d))
        if side * side != d:
            raise RuntimeError(f"Dimension NB incompatible avec une image carrée : d={d}")
        rgb = image.convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
        values = np.asarray(rgb, dtype=np.float64).mean(axis=2) / 255.0
    return np.ascontiguousarray(values.reshape(d).astype(dtype))


def linear_scores(x: np.ndarray, model: dict[str, object]) -> np.ndarray:
    d = int(model["d"])
    W = np.asarray(model["W"], dtype=np.float64).reshape(3, d)
    b = np.asarray(model["b"], dtype=np.float64)
    return W @ x.astype(np.float64) + b


def mlp_probabilities(x: np.ndarray, model: dict[str, object]) -> np.ndarray:
    d, H = int(model["d"]), int(model["H"])
    W1 = np.asarray(model["W1"], dtype=np.float64).reshape(H, d)
    b1 = np.asarray(model["b1"], dtype=np.float64)
    W2 = np.asarray(model["W2"], dtype=np.float64).reshape(3, H)
    b2 = np.asarray(model["b2"], dtype=np.float64)
    z1 = W1 @ x.astype(np.float64) + b1
    hidden = 1.0 / (1.0 + np.exp(-np.clip(z1, -700.0, 700.0)))
    logits = W2 @ hidden + b2
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def rbf_scores(x: np.ndarray, model: dict[str, object]) -> np.ndarray:
    d = int(model["d"])
    nb_centres = int(model["nb_centres"])
    gamma = float(model["gamma"])
    centres = np.asarray(model["centres"], dtype=np.float64).reshape(nb_centres, d)
    W = np.asarray(model["W"], dtype=np.float64).reshape(3, nb_centres)
    distances = np.sum((centres - x.astype(np.float64)) ** 2, axis=1)
    influences = np.exp(-gamma * distances)
    b = np.asarray(model["b"], dtype=np.float64)
    return W @ influences + b


def prediction_details(
    model_name: str,
    values: np.ndarray,
    model: dict[str, object],
    already_loaded: bool,
) -> str:
    if already_loaded:
        loading_note = "modèle déjà chargé en mémoire"
    elif bool(model.get("from_cache")):
        loading_note = "paramètres rechargés depuis le cache"
    else:
        loading_note = "modèle entraîné puis sauvegardé dans le cache"

    if model_name == "MLP":
        metric_name = "Probabilité softmax interne"
        rows = "\n".join(
            f"| {LABELS[index]} | {float(value) * 100:.2f} % |"
            for index, value in enumerate(values)
        )
        warning = (
            "Ces valeurs proviennent directement de la couche softmax du MLP. "
            "Elles totalisent 100 %, mais ne constituent pas des probabilités calibrées."
        )
    else:
        metric_name = "Score brut"
        rows = "\n".join(
            f"| {LABELS[index]} | {float(value):.6g} |"
            for index, value in enumerate(values)
        )
        warning = (
            "Ces scores servent uniquement à classer les trois sorties : la valeur la plus élevée "
            "détermine la prédiction. Ils ne doivent pas être interprétés comme des probabilités."
        )

    return (
        f"**{model_name}** · {'RGB' if model_name == 'RBF' else 'NB'} normalisé · {loading_note}\n\n"
        f"| Classe | {metric_name} |\n|---|---:|\n{rows}\n\n"
        f"*{warning}*"
    )


def predict(image: Image.Image | None, model_name: str) -> tuple[str, str]:
    if image is None:
        raise gr.Error("Ajoutez d'abord une image.")
    try:
        model, already_loaded = get_model(model_name)
        d = int(model["d"])
        representation = "rgb" if model_name == "RBF" else "nb"
        x = preprocess_image(image, d, np.float64, representation)

        if model_name == "Modèle linéaire":
            values = linear_scores(x, model)
        elif model_name == "MLP":
            values = mlp_probabilities(x, model)
        else:
            values = rbf_scores(x, model)

        predicted_class = int(np.argmax(values))
        details = prediction_details(model_name, values, model, already_loaded)
        return LABELS[predicted_class], details
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
.hero { margin: 12px 0 28px; }
.hero h1 { font-size: 2.2rem; margin-bottom: 8px; }
.hero p { color: #6b6b68; max-width: 720px; }
.result-box textarea { font-size: 1.25rem !important; font-weight: 650 !important; }
"""

with gr.Blocks(title="État de la route") as demo:
    gr.HTML("""
    <section class="hero">
      <h1>Reconnaissance de l’état d’une route</h1>
      <p>Chargez une photographie, choisissez un modèle puis lancez la prédiction.
      Le prototype distingue une route sèche, mouillée ou enneigée.</p>
    </section>
    """)
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            image_input = gr.Image(type="pil", label="Photographie de la route")
        with gr.Column(scale=2):
            model_input = gr.Dropdown(
                choices=list(MODEL_CONFIG), value="Modèle linéaire", label="Modèle"
            )
            predict_button = gr.Button("Analyser l’image", variant="primary")
            result_output = gr.Textbox(
                label="Prédiction", interactive=False, elem_classes="result-box"
            )
            details_output = gr.Markdown()

    predict_button.click(
        fn=predict,
        inputs=[image_input, model_input],
        outputs=[result_output, details_output],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
