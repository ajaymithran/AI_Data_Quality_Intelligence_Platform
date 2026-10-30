"""Initialize the project: create ``.venv`` and install dependencies.

Usage::

    python init.py

Then start the interface with the command it prints, e.g.::

    .venv\\Scripts\\python run.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
MIN_PYTHON = (3, 10)


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        print(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
            f"(found {sys.version.split()[0]})."
        )
        return 1

    if not VENV.exists():
        print(f"Creating virtual environment in {VENV} ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    else:
        print(f"Reusing existing virtual environment in {VENV}")

    python = _venv_python()
    if not python.exists():
        print(f"Could not find the virtualenv interpreter at {python}.")
        return 1

    print("Upgrading pip ...")
    subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing dependencies ...")
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    )

    print()
    print("Setup complete.")
    print("Start the interface with:")
    if os.name == "nt":
        print(r"  .venv\Scripts\python run.py")
    else:
        print("  .venv/bin/python run.py")
    print("Then open http://localhost:3000/universal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
