from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

app = FastAPI(title="HomAIker Inference API")
_executor = ThreadPoolExecutor(max_workers=4)

_static_dir = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(content=(_static_dir / "index.html").read_text(encoding="utf-8"), status_code=200)


@app.get("/api/metrics")
async def get_metrics() -> dict:
    """Return evaluation metrics (LoRA vs Base) from evaluation/results.json if present."""
    metrics_path = _project_root / "evaluation" / "results.json"
    if metrics_path.exists():
        import json
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            return {
                "lora": data.get("lora_aggregate", {}),
                "base": data.get("base_aggregate", {}),
                "improvement_pct": data.get("improvement_pct", {}),
                "n_samples": data.get("n_samples", 0),
            }
        except Exception:
            pass
    return {"lora": {}, "base": {}, "improvement_pct": {}, "n_samples": 0}


@app.get("/api/config")
async def get_config() -> dict:
    return {
        "default_lora_url": os.getenv("LORA_URL", ""),
        "default_prompt": os.getenv("DEFAULT_PROMPT", ""),
    }


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    lora_url: str = Form(...),
    lora_scale: float = Form(1.0),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(2.5),
    seed: str = Form(""),
    output_format: str = Form("png"),
    enable_prompt_expansion: bool = Form(False),
    acceleration: str = Form("regular"),
) -> JSONResponse:
    if not lora_url.strip():
        raise HTTPException(status_code=400, detail="lora_url is required")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    seed_int = None
    if seed.strip().lstrip("-").isdigit():
        seed_int = int(seed.strip())

    image_bytes = await image.read()
    suffix = ".png"
    if image.filename and "." in image.filename:
        suffix = "." + image.filename.rsplit(".", 1)[-1].lower()

    def _run() -> list:
        from inference.fal_inference import edit_with_lora_flux2

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        try:
            return edit_with_lora_flux2(
                image_path=tmp_path,
                prompt=prompt,
                lora_path_or_url=lora_url,
                lora_scale=lora_scale,
                num_images=1,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed_int,
                output_format=output_format,
                enable_prompt_expansion=enable_prompt_expansion,
                acceleration=acceleration,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    loop = asyncio.get_event_loop()
    try:
        images = await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not images:
        raise HTTPException(status_code=500, detail="No images returned from fal.ai")

    return JSONResponse({
        "images": [
            {"url": img.url, "width": img.width, "height": img.height}
            for img in images
        ]
    })


app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
