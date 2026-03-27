#!/usr/bin/env python
"""Start the HomAIker frontend server and optionally expose it via ngrok.

Run from the project root:
    python frontend/start.py

Environment variables (from .env or shell):
    PORT              Local port (default 8000)
    NGROK_AUTHTOKEN   ngrok auth token (required for automatic tunnel)
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

PORT = int(os.getenv("PORT", "8000"))
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")


def _start_ngrok(port: int) -> None:
    try:
        from pyngrok import conf, ngrok
    except ImportError:
        print(
            "[ngrok] pyngrok not installed. Install it with:\n"
            "    pip install pyngrok\n"
            "[ngrok] Skipping tunnel — server is available at "
            f"http://localhost:{port}"
        )
        return

    if NGROK_AUTHTOKEN:
        conf.get_default().auth_token = NGROK_AUTHTOKEN
    else:
        print(
            "[ngrok] NGROK_AUTHTOKEN not set. "
            "Add it to .env or run `ngrok config add-authtoken <token>`."
        )

    try:
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        print("\n" + "=" * 60)
        print(f"  Public URL : {public_url}")
        print(f"  Local URL  : http://localhost:{port}")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"[ngrok] Failed to start tunnel: {exc}")
        print(f"[ngrok] Server still available at http://localhost:{port}")


def main() -> None:
    import uvicorn

    print(f"Starting HomAIker server on http://localhost:{PORT} ...")

    def _ngrok_thread() -> None:
        time.sleep(2)
        _start_ngrok(PORT)

    t = threading.Thread(target=_ngrok_thread, daemon=True)
    t.start()

    uvicorn.run(
        "frontend.server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
