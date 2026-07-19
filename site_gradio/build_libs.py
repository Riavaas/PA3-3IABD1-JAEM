from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = platform.system()


def compiler(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"Compilateur introuvable : {name}. "
            "Sous Windows, installez GCC/G++ avec MSYS2 ou MinGW-w64 puis ajoutez-les au PATH."
        )
    return path


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def output_name(stem: str) -> str:
    if SYSTEM == "Windows":
        return f"{stem}.dll"
    if SYSTEM == "Darwin":
        return f"{stem}.dylib"
    return f"{stem}.so"


def shared_flags(cpp: bool = False) -> list[str]:
    if SYSTEM == "Darwin":
        return ["-O2", "-dynamiclib"]
    flags = ["-O2", "-shared"]
    if SYSTEM != "Windows":
        flags.append("-fPIC")
    elif cpp:
        # Évite de dépendre de libstdc++-6.dll et libgcc_s_*.dll au chargement.
        flags.extend(["-static-libgcc", "-static-libstdc++"])
    else:
        flags.append("-static-libgcc")
    return flags


def main() -> None:
    gcc = compiler("gcc")
    gpp = compiler("g++")

    eigen = ROOT / "models" / "rbf" / "eigen-5.0.0" / "Eigen" / "Dense"
    if not eigen.exists():
        raise SystemExit(
            "Eigen 5.0.0 est absent : models/rbf/eigen-5.0.0/Eigen/Dense"
        )

    run([
        gcc, *shared_flags(),
        "models/lineaire/linear_model_lib.c",
        "-o", f"models/lineaire/{output_name('linear_model_lib')}",
    ])
    run([
        gcc, *shared_flags(),
        "models/mlp/mlp_lib.c",
        "-o", f"models/mlp/{output_name('mlp_lib')}",
        "-lm",
    ])
    run([
        gpp, *shared_flags(cpp=True), "-std=c++17",
        "models/rbf/rbf_lib.cpp",
        "-o", f"models/rbf/{output_name('rbf_lib')}",
    ])

    print(f"Bibliothèques compilées pour {SYSTEM}.")


if __name__ == "__main__":
    main()
