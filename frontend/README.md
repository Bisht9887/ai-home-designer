# HomAIker Frontend

A web UI for running inference on the finetuned **FLUX.2 LoRA** model.  
The server wraps `inference.fal_inference.edit_with_lora_flux2` and delegates
all GPU work to [fal.ai](https://fal.ai) — the FastAPI server itself is lightweight.

---

## Architecture

```
Browser → HTTPS → FastAPI (frontend/server.py) → fal.ai (FLUX.2 LoRA)
```

- **`server.py`** — FastAPI app; serves the React SPA + `/api/generate` endpoint
- **`static/index.html`** — React SPA (CDN-based, no build step required)
- **`start.py`** — local dev helper: starts uvicorn + optional ngrok tunnel

---

## 1) Deploy to Render (recommended)

The app is configured for [Render](https://render.com) via `render.yaml` at the project root.

### Steps
1. Push the repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Blueprint** → connect the repo.
3. Render detects `render.yaml` automatically and creates the **homaiker** web service.
4. In the Render dashboard set the following **environment variables**:

| Variable | Description |
|---|---|
| `FAL_KEY` | Your fal.ai API key |
| `LORA_URL` | CDN URL of `diffusers_lora.safetensors` (from fal.ai storage) |
| `DEFAULT_PROMPT` | Optional: pre-fill prompt in the UI |

5. Click **Deploy** — Render gives you a permanent `https://<name>.onrender.com` URL.

**Evaluation metrics on UI:** To show LPIPS, SSIM, MSE (LoRA vs Base) in the UI, run:
```bash
python -m evaluation.run_eval --flat-dir evaluation/eval_1 --compare --output-json evaluation/results.json
```
The frontend reads `evaluation/results.json` via `/api/metrics`.

> **Free tier note:** Render's free plan spins down after 15 min of inactivity (cold start ~30s on first request). Upgrade to a paid plan for always-on.

---

## 2) Local setup

### Install requirements

From the repo root:

```bash
pip install -r frontend/requirements.txt
```

### Configure environment

Copy `.env.example` to the project root `.env` (it already exists) and add:

```env
FAL_KEY=your_fal_key_here

# URL to your downloaded LoRA weights on fal.ai CDN
LORA_URL=https://fal.run/files/<your-lora-file-url>

# Optional: pre-fill the prompt in the UI
DEFAULT_PROMPT=Furnish this floor plan with modern Scandinavian style

# ngrok auth token (https://dashboard.ngrok.com/get-started/your-authtoken)
NGROK_AUTHTOKEN=your_ngrok_authtoken_here

# Port (default 8000)
PORT=8000
```

The `LORA_URL` is the CDN URL of `diffusers_lora.safetensors` downloaded by
`finetuning.cli wait`. You can also paste any `https://` URL directly in the UI.

---

## 3) Run locally

```bash
python frontend/start.py
```

This will:
1. Start the FastAPI server on `http://localhost:8000`
2. Open an ngrok tunnel and print the **public URL**

Open the printed URL in any browser to use the app.

---

## 4) Run without ngrok (local only)

```bash
uvicorn frontend.server:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

---

## 5) Manual ngrok (alternative)

Start the server separately:

```bash
uvicorn frontend.server:app --host 0.0.0.0 --port 8000
```

In another terminal, start ngrok manually:

```bash
ngrok http 8000
```

---

## 6) Usage

1. **Upload** a floor plan image (PNG, JPEG, or WEBP) by drag-and-drop or click.
2. **Enter a prompt** describing the desired interior style.
3. **LoRA Weights URL** — pre-filled from `LORA_URL` in `.env`; can be overridden.
4. Optionally expand **Advanced Settings** to tune LoRA scale, steps, guidance, seed, etc.
5. Click **Generate Interior** — the result appears on the right and can be downloaded.

---

## 7) API reference

### `GET /api/config`
Returns server-configured defaults:
```json
{ "default_lora_url": "...", "default_prompt": "..." }
```

### `POST /api/generate`
Multipart form fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | required | Input floor plan image |
| `prompt` | string | required | Edit description |
| `lora_url` | string | required | LoRA weights URL |
| `lora_scale` | float | `1.0` | LoRA influence strength |
| `num_inference_steps` | int | `28` | Diffusion steps |
| `guidance_scale` | float | `2.5` | Guidance scale |
| `seed` | string | `""` | Empty = random |
| `acceleration` | string | `"regular"` | `none` / `regular` / `high` |
| `enable_prompt_expansion` | bool | `false` | Expand prompt with fal |

Response:
```json
{
  "images": [{ "url": "https://...", "width": 1024, "height": 1024 }]
}
```
