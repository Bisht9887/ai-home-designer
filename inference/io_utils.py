from __future__ import annotations

from pathlib import Path
import time

import requests


def download_file(url: str, output_path: Path, *, timeout: float = 120.0) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        last_print = time.time()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print >= 2.0:
                        if total_bytes:
                            pct = (downloaded / total_bytes) * 100
                            print(f"Downloading {output_path.name}: {downloaded/1e6:.1f}MB / {total_bytes/1e6:.1f}MB ({pct:.1f}%)")
                        else:
                            print(f"Downloading {output_path.name}: {downloaded/1e6:.1f}MB")
                        last_print = now
        if total_bytes:
            print(f"Downloaded {output_path.name}: {downloaded/1e6:.1f}MB")
    return output_path
