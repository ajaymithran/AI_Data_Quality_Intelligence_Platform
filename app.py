"""Standalone ASGI app exposing the universal data-quality scanner UI.

The interface is available at ``/universal`` (and ``/`` for convenience).
Everything else lives under ``/api/universal/...``.

Run it with::

    python run.py            # http://localhost:3000/universal
    # or
    uvicorn app:app --port 3000
"""

from universal.web import build_app

app = build_app()


@app.get("/health")
def health() -> dict:
    """Small liveness probe."""
    return {"status": "ok", "interface": "/universal"}
