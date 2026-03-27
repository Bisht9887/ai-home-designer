from __future__ import annotations

import json
import os
import time
from urllib.parse import urlencode, urlparse, urlunparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import requests


ApplicationId = Literal["fal-ai/flux-2-trainer/edit"]


@dataclass(frozen=True)
class TrainRequest:
    request_id: str


@dataclass(frozen=True)
class TrainResult:
    diffusers_lora_url: str
    config_url: str


def _extract_fal_detail_from_http_error(err: requests.HTTPError) -> Optional[str]:
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    try:
        body = resp.json()
    except Exception:
        return None
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
    return None


def _normalize_queue_status(status: Any) -> Optional[str]:
    if isinstance(status, dict):
        s = status.get("status")
        return str(s) if s is not None else None

    s = getattr(status, "status", None)
    if s is not None:
        return str(s)

    s = getattr(status, "state", None)
    if s is not None:
        return str(s)

    name = getattr(getattr(status, "__class__", None), "__name__", None)
    if not name:
        return None

    upper = name.upper()
    if upper == "COMPLETED":
        return "COMPLETED"
    if upper == "INPROGRESS":
        return "IN_PROGRESS"
    if upper == "QUEUED":
        return "IN_QUEUE"

    return upper


def _get_response_url(status: Any) -> Optional[str]:
    if isinstance(status, dict):
        url = status.get("response_url")
        return str(url) if url else None
    url = getattr(status, "response_url", None)
    return str(url) if url else None


def _get_status_url(status: Any) -> Optional[str]:
    if isinstance(status, dict):
        url = status.get("status_url")
        return str(url) if url else None
    url = getattr(status, "status_url", None)
    return str(url) if url else None


def _queue_get_json(
    url: str,
    *,
    auth_header_value: str,
    timeout: float = 120.0,
    allow_post_fallback: bool = False,
    post_json: Optional[dict] = None,
) -> Any:
    headers = {
        "Authorization": auth_header_value,
        "Accept": "application/json",
        "User-Agent": "homAIker-finetuning/1.0",
    }

    def _do_get() -> requests.Response:
        return requests.get(url, headers=headers, timeout=timeout)

    def _do_post() -> requests.Response:
        # Some environments route queue endpoints as POST-only.
        if post_json is None:
            return requests.post(url, headers=headers, timeout=timeout)
        return requests.post(
            url,
            headers={**headers, "Content-Type": "application/json"},
            json=post_json,
            timeout=timeout,
        )

    resp = _do_get()
    if resp.status_code == 405:
        allow = (resp.headers.get("Allow") or "").upper()
        if allow_post_fallback and "POST" in allow:
            resp = _do_post()
        else:
            body = resp.text
            if len(body) > 500:
                body = body[:500] + "..."
            raise requests.HTTPError(
                f"405 Method Not Allowed for url: {url} Allow={resp.headers.get('Allow')} Body={body!r}",
                response=resp,
            )
    if resp.status_code >= 400:
        body = resp.text
        if len(body) > 500:
            body = body[:500] + "..."
        allow = resp.headers.get("Allow")
        raise requests.HTTPError(
            f"{resp.status_code} Client Error for url: {url} Allow={allow} Body={body!r}",
            response=resp,
        )
    resp.raise_for_status()
    return resp.json()


def _queue_request_json(
    url: str,
    *,
    auth_header_value: str,
    method: Literal["GET", "POST"],
    timeout: float = 120.0,
    post_json: Optional[dict] = None,
) -> Any:
    """Issue a specific method request to the queue API and parse JSON."""

    headers = {
        "Authorization": auth_header_value,
        "Accept": "application/json",
        "User-Agent": "homAIker-finetuning/1.0",
    }

    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=timeout)
    else:
        if post_json is None:
            resp = requests.post(url, headers=headers, timeout=timeout)
        else:
            resp = requests.post(
                url,
                headers={**headers, "Content-Type": "application/json"},
                json=post_json,
                timeout=timeout,
            )

    if resp.status_code >= 400:
        body = resp.text
        if len(body) > 500:
            body = body[:500] + "..."
        raise requests.HTTPError(
            f"{resp.status_code} Client Error for url: {url} Body={body!r}",
            response=resp,
        )

    return resp.json()


def _with_query(url: str, **params: Any) -> str:
    parsed = urlparse(url)
    existing = parsed.query
    q = {}
    if existing:
        # Keep existing query if present; status_url from API is unlikely to include one.
        q_str = existing.split("&")
        for part in q_str:
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                q[k] = v
            else:
                q[part] = ""
    for k, v in params.items():
        q[k] = str(v)
    new_query = urlencode(q)
    return urlunparse(parsed._replace(query=new_query))


def _repair_response_url(response_url: str, *, application: str, request_id: str) -> str:
    """Normalize response URLs to include the configured application path.

    In some environments the queue may return a response_url that omits parts of the
    endpoint id (e.g. missing `/edit`). We repair it by forcing the path to:
    `/{application}/requests/{request_id}`.
    """

    parsed = urlparse(response_url)
    expected_path = f"/{application}/requests/{request_id}"

    if parsed.netloc == "queue.fal.run" and parsed.path != expected_path:
        return urlunparse(parsed._replace(path=expected_path, query=""))

    return response_url


def _normalize_application_in_url(url: str, *, application: str) -> str:
    """Fix queue URLs that drop the last path segment (e.g. missing `/edit`)."""

    parsed = urlparse(url)
    if parsed.netloc != "queue.fal.run":
        return url

    app_no_suffix = application
    if app_no_suffix.endswith("/edit"):
        app_no_suffix = app_no_suffix[: -len("/edit")]

    expected_prefix = f"/{application}/requests/"
    wrong_prefix = f"/{app_no_suffix}/requests/"
    if parsed.path.startswith(wrong_prefix) and not parsed.path.startswith(expected_prefix):
        new_path = expected_prefix + parsed.path[len(wrong_prefix) :]
        return urlunparse(parsed._replace(path=new_path))

    return url


def _get_request_id(handle: Any) -> str:
    if isinstance(handle, dict):
        for key in ("request_id", "requestId"):
            if key in handle:
                return str(handle[key])
    for attr in ("request_id", "requestId"):
        if hasattr(handle, attr):
            return str(getattr(handle, attr))
    raise ValueError(f"Could not extract request_id from handle: {handle!r}")


def upload_training_zip(zip_path: Path) -> str:
    return upload_file(zip_path)


def upload_file(path: Path) -> str:
    try:
        import fal_client

        if hasattr(fal_client, "upload_file"):
            return fal_client.upload_file(str(path))

        client = fal_client.SyncClient()
        return client.upload_file(str(path))
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "fal_client is required. Install it with: pip install fal-client"
        ) from e


def submit_edit_trainer(
    *,
    image_data_url: str,
    steps: int = 1000,
    learning_rate: float = 5e-5,
    default_caption: Optional[str] = None,
    network_weights: Optional[str] = None,
    output_lora_format: Literal["fal", "comfy"] = "fal",
    application: ApplicationId = "fal-ai/flux-2-trainer/edit",
) -> TrainRequest:
    import fal_client

    arguments: Dict[str, Any] = {
        "image_data_url": image_data_url,
        "steps": steps,
        "learning_rate": learning_rate,
        "default_caption": default_caption,
        "output_lora_format": output_lora_format,
    }

    if network_weights is not None:
        arguments["network_weights"] = network_weights

    handle = fal_client.submit(application, arguments)
    return TrainRequest(request_id=_get_request_id(handle))


def poll_until_complete(
    request: TrainRequest,
    *,
    poll_seconds: float = 10.0,
    with_logs: bool = True,
    application: ApplicationId = "fal-ai/flux-2-trainer/edit",
    timeout_seconds: Optional[float] = None,
) -> TrainResult:
    import fal_client

    start = time.time()
    last_status: Any = None
    last_printed_status: Optional[str] = None
    _server_error_retries = 0
    _max_server_error_retries = 10
    while True:
        try:
            status = fal_client.status(application, request.request_id, with_logs=with_logs)
            _server_error_retries = 0
        except Exception as _exc:
            _code = getattr(_exc, "status_code", None) or getattr(_exc, "status", None)
            if _code is None:
                _cause = getattr(_exc, "__cause__", None)
                _code = getattr(_cause, "status_code", None) or getattr(_cause, "status", None)
            if _code is not None and 500 <= int(_code) < 600:
                _server_error_retries += 1
                if _server_error_retries <= _max_server_error_retries:
                    print(
                        f"[warn] fal.ai returned {_code} (attempt {_server_error_retries}/{_max_server_error_retries}), retrying in {poll_seconds}s…",
                        flush=True,
                    )
                    time.sleep(poll_seconds)
                    continue
                raise EnvironmentError(
                    f"fal.ai returned {_code} for request {request.request_id} on every attempt "
                    f"({_max_server_error_retries} retries). The job likely failed or was cancelled "
                    f"on fal.ai's side.\n"
                    f"  • Check the job in the fal.ai dashboard: https://fal.ai/dashboard\n"
                    f"  • Submit a new training job: python -m finetuning.cli train <zip> --steps <N>"
                ) from _exc
            raise
        last_status = status
        status_str = _normalize_queue_status(status)

        if status_str and status_str != last_printed_status:
            print(f"Training request status: {status_str}", flush=True)
            last_printed_status = status_str

        if status_str == "COMPLETED":
            break

        if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
            raise TimeoutError(f"Training did not complete within {timeout_seconds} seconds")

        time.sleep(poll_seconds)

    fal_key = os.getenv("FAL_KEY")
    auth_header_value: Optional[str] = None
    try:
        from fal_client.auth import fetch_auth_credentials

        auth_header_value = fetch_auth_credentials().header_value
    except Exception:
        auth_header_value = None

    if auth_header_value is None:
        if not fal_key:
            raise EnvironmentError("FAL_KEY is not set")
        auth_header_value = fal_key if fal_key.lower().startswith("key ") else f"Key {fal_key}"

    # Result retrieval flow for this endpoint in this environment:
    # - POST /edit/requests/{id}/status returns an output-handle (new request_id) under
    #   /fal-ai/flux-2-trainer/requests/{new_id}
    # - GET that handle's status_url until COMPLETED
    # - then GET that handle's response_url to obtain the final output payload

    # If you already have an output-handle request id (e.g. printed by a previous run),
    # you can resume without POSTing again. This is useful if your account becomes locked
    # for billing reasons after the handle was minted.
    resume_handle_request_id = os.getenv("FAL_OUTPUT_HANDLE_REQUEST_ID")
    if resume_handle_request_id:
        resume_handle_request_id = resume_handle_request_id.strip()

    if resume_handle_request_id:
        print(f"Resuming with output handle request_id from FAL_OUTPUT_HANDLE_REQUEST_ID: {resume_handle_request_id}", flush=True)
        handle_payload = {
            "request_id": resume_handle_request_id,
            "status_url": f"https://queue.fal.run/fal-ai/flux-2-trainer/requests/{resume_handle_request_id}/status",
            "response_url": f"https://queue.fal.run/fal-ai/flux-2-trainer/requests/{resume_handle_request_id}",
        }
    else:
        # fal_client strips the path component (e.g. /edit) when constructing queue URLs,
        # so the correct base for result retrieval is always owner/alias without the path.
        _app_parts = application.split("/")
        _app_base = "/".join(_app_parts[:2])  # e.g. "fal-ai/flux-2-trainer"
        status_post_url = f"https://queue.fal.run/{_app_base}/requests/{request.request_id}/status?logs=0"
        print("Requesting output handle...", flush=True)
        try:
            handle_payload = _queue_request_json(
                status_post_url,
                auth_header_value=auth_header_value,
                method="POST",
                post_json={},
            )
        except requests.HTTPError as e:
            detail = _extract_fal_detail_from_http_error(e)
            if detail and "user is locked" in detail.lower():
                raise EnvironmentError(
                    "Fal account is locked (exhausted balance). Top up/unlock at "
                    "https://fal.ai/dashboard/billing and rerun. "
                    "If you already saw an output-handle request_id printed earlier (e.g. 90054a28-...), "
                    "you can set FAL_OUTPUT_HANDLE_REQUEST_ID=<that-id> to resume without POSTing again. "
                    f"Detail: {detail}"
                ) from e
            raise

    if not isinstance(handle_payload, dict) or "status_url" not in handle_payload or "response_url" not in handle_payload:
        raise ValueError(
            "Unexpected status handle payload: "
            f"{json.dumps(handle_payload, indent=2) if isinstance(handle_payload, dict) else handle_payload!r}"
        )

    handle_status_url = str(handle_payload["status_url"])
    handle_response_url = str(handle_payload["response_url"])
    handle_request_id = str(handle_payload.get("request_id", ""))
    if handle_request_id:
        print(f"Output handle request_id: {handle_request_id}", flush=True)
    print(f"Output handle status_url: {handle_status_url}", flush=True)
    print(f"Output handle response_url: {handle_response_url}", flush=True)

    # Poll handle status until COMPLETED
    handle_start = time.time()
    last_handle_state: Optional[str] = None
    last_heartbeat: float = 0.0
    while True:
        handle_status_payload = _queue_request_json(
            handle_status_url,
            auth_header_value=auth_header_value,
            method="GET",
        )
        handle_state = _normalize_queue_status(handle_status_payload)
        if handle_state and handle_state != last_handle_state:
            print(f"Output handle status: {handle_state}", flush=True)
            last_handle_state = handle_state
        now = time.time()
        if now - last_heartbeat >= max(poll_seconds, 30.0):
            elapsed = int(now - handle_start)
            if handle_state:
                print(f"Still waiting on output handle ({elapsed}s): {handle_state}", flush=True)
            else:
                print(f"Still waiting on output handle ({elapsed}s)...", flush=True)
            last_heartbeat = now
        if handle_state == "COMPLETED":
            break
        if timeout_seconds is not None and (time.time() - handle_start) > timeout_seconds:
            raise TimeoutError(f"Result handle did not complete within {timeout_seconds} seconds")
        time.sleep(poll_seconds)

    # Fetch final output. GET may return 400 "Request is still in progress" briefly.
    headers = {"Authorization": auth_header_value, "Accept": "application/json"}
    result_start = time.time()
    printed_in_progress = False
    while True:
        resp = requests.get(handle_response_url, headers=headers, timeout=120.0)
        if resp.status_code == 200:
            result = resp.json()
            break
        if resp.status_code == 400:
            try:
                body = resp.json()
            except Exception:
                body = {"detail": resp.text}
            detail = str(body.get("detail", "")) if isinstance(body, dict) else ""
            if "in progress" in detail.lower():
                if not printed_in_progress:
                    print("Finalizing output (still in progress)...", flush=True)
                    printed_in_progress = True
                if timeout_seconds is not None and (time.time() - result_start) > timeout_seconds:
                    raise TimeoutError(f"Result did not become available within {timeout_seconds} seconds")
                time.sleep(poll_seconds)
                continue
        body_txt = resp.text
        if len(body_txt) > 500:
            body_txt = body_txt[:500] + "..."
        # 404 here is a known fal.ai bug: the handle worker fails internally with
        # UNAUTHENTICATED when fetching the edit-endpoint result.  Fall back to
        # extracting the artifact URLs that the trainer logged directly.
        if resp.status_code == 404:
            print(
                "Output handle returned 404 (known fal.ai issue). "
                "Falling back to log-based URL extraction…",
                flush=True,
            )
            log_result = _extract_urls_from_training_logs(application, request.request_id)
            if log_result is not None:
                return log_result
        raise requests.HTTPError(
            f"{resp.status_code} Client Error for url: {handle_response_url} Body={body_txt!r}",
            response=resp,
        )

    def _extract_url(obj: Any) -> str:
        if isinstance(obj, dict) and "url" in obj:
            return str(obj["url"])
        raise ValueError(f"Unexpected file object: {obj!r}")

    diffusers = result.get("diffusers_lora_file") if isinstance(result, dict) else None
    config = result.get("config_file") if isinstance(result, dict) else None
    if diffusers is None or config is None:
        raise ValueError(f"Unexpected result payload: {json.dumps(result, indent=2) if isinstance(result, dict) else result!r}")

    return TrainResult(
        diffusers_lora_url=_extract_url(diffusers),
        config_url=_extract_url(config),
    )


def _extract_urls_from_training_logs(
    application: str,
    request_id: str,
) -> Optional["TrainResult"]:
    """Parse artifact URLs from training job logs as a fallback when the output handle fails."""
    import re
    import fal_client

    try:
        status = fal_client.status(application, request_id, with_logs=True)
    except Exception as exc:
        print(f"  [warn] Could not fetch training logs for URL extraction: {exc}", flush=True)
        return None

    logs = getattr(status, "logs", None) or []
    lora_url: Optional[str] = None
    config_url: Optional[str] = None
    url_pattern = re.compile(r"url='(https?://[^']+)'")

    for entry in logs:
        msg = entry.get("message", "") if isinstance(entry, dict) else str(entry)
        m = url_pattern.search(msg)
        if not m:
            continue
        url = m.group(1)
        if "result" in msg.lower() or "lora" in msg.lower() or "safetensors" in url:
            lora_url = url
        elif "config" in msg.lower() or "config" in url:
            config_url = url

    if lora_url and config_url:
        print(f"  Extracted LoRA URL from logs:   {lora_url}", flush=True)
        print(f"  Extracted config URL from logs: {config_url}", flush=True)
        return TrainResult(diffusers_lora_url=lora_url, config_url=config_url)

    print(
        f"  [warn] Could not extract both URLs from logs "
        f"(lora={lora_url!r}, config={config_url!r})",
        flush=True,
    )
    return None


def download_file(url: str, output_path: Path, *, timeout: float = 120.0) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting download: {output_path.name}")
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
        else:
            print(f"Downloaded {output_path.name}: {downloaded/1e6:.1f}MB")
    return output_path
