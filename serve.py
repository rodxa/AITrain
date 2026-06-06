from __future__ import annotations

import os

from waitress import serve

from app import app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    serve(app, host=host, port=port)
