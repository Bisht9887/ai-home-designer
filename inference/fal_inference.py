from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


TextToImageApp = Literal["fal-ai/flux/dev", "fal-ai/flux-lora"]
ImageToImageApp = Literal["fal-ai/flux/dev/image-to-image", "fal-ai/flux-lora/image-to-image", "fal-ai/flux-2/lora/edit"]


@dataclass(frozen=True)
class GeneratedImage:
    url: str
    width: Optional[int] = None
    height: Optional[int] = None


def upload_file(path: Path) -> str:
    try:
        import fal_client

        if hasattr(fal_client, "upload_file"):
            return fal_client.upload_file(str(path))

        client = fal_client.SyncClient()
        return client.upload_file(str(path))
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("fal_client is required. Install it with: pip install fal-client") from e


def text_to_image_base(
    *,
    prompt: str,
    num_images: int = 1,
    image_size: str = "landscape_4_3",
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
    output_format: str = "png",
    app: TextToImageApp = "fal-ai/flux/dev",
) -> List[GeneratedImage]:
    import fal_client

    arguments: Dict[str, Any] = {
        "prompt": prompt,
        "num_images": num_images,
        "image_size": image_size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "output_format": output_format,
    }

    handle = fal_client.submit(app, arguments)
    request_id = getattr(handle, "request_id", None) or getattr(handle, "requestId", None) or handle.get("request_id")
    result = fal_client.result(app, request_id)

    images = []
    for img in result.get("images", []):
        images.append(GeneratedImage(url=img.get("url"), width=img.get("width"), height=img.get("height")))
    return images


def image_to_image_base(
    *,
    image_path: Path,
    prompt: str,
    strength: float = 0.95,
    num_images: int = 1,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
    output_format: str = "png",
    app: ImageToImageApp = "fal-ai/flux/dev/image-to-image",
) -> List[GeneratedImage]:
    import fal_client

    image_url = upload_file(image_path)
    arguments: Dict[str, Any] = {
        "image_url": image_url,
        "prompt": prompt,
        "strength": strength,
        "num_images": num_images,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "output_format": output_format,
    }

    handle = fal_client.submit(app, arguments)
    request_id = getattr(handle, "request_id", None) or getattr(handle, "requestId", None) or handle.get("request_id")
    result = fal_client.result(app, request_id)

    images = []
    for img in result.get("images", []):
        images.append(GeneratedImage(url=img.get("url"), width=img.get("width"), height=img.get("height")))
    return images


def edit_with_lora_flux2(
    *,
    image_path: Path,
    prompt: str,
    lora_path_or_url: str,
    lora_scale: float = 1.0,
    num_images: int = 1,
    num_inference_steps: int = 28,
    guidance_scale: float = 2.5,
    seed: Optional[int] = None,
    output_format: str = "png",
    enable_prompt_expansion: bool = False,
    acceleration: str = "regular",
    app: ImageToImageApp = "fal-ai/flux-2/lora/edit",
) -> List[GeneratedImage]:
    import fal_client

    image_url = upload_file(image_path)

    if lora_path_or_url.startswith("http://") or lora_path_or_url.startswith("https://"):
        lora_url = lora_path_or_url
    else:
        lora_url = upload_file(Path(lora_path_or_url))

    arguments: Dict[str, Any] = {
        "prompt": prompt,
        "image_urls": [image_url],
        "loras": [{"path": lora_url, "scale": lora_scale}],
        "num_images": num_images,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "output_format": output_format,
        "enable_prompt_expansion": enable_prompt_expansion,
        "acceleration": acceleration,
    }

    handle = fal_client.submit(app, arguments)
    request_id = getattr(handle, "request_id", None) or getattr(handle, "requestId", None) or handle.get("request_id")
    result = fal_client.result(app, request_id)

    images = []
    for img in result.get("images", []):
        images.append(GeneratedImage(url=img.get("url"), width=img.get("width"), height=img.get("height")))
    return images
