"""Start the data_qualiy v1 interface.

Usage::

    .venv\\Scripts\\python run.py      # Windows
    .venv/bin/python run.py           # macOS / Linux

Then open http://localhost:3000/universal
"""

from __future__ import annotations

import uvicorn

from app import app

HOST = "127.0.0.1"
PORT = 3000


def main() -> None:
    print(f"Universal data-quality scanner -> http://localhost:{PORT}/universal")
    print("Press CTRL+C to stop.")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
