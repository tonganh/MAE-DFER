from pathlib import Path

import httpx
from fastapi import HTTPException, UploadFile

from .config import httpx_trust_env
from .video_io import ALLOWED_SUFFIX


async def forward_video_multipart_to_url(url: str, file: UploadFile) -> dict:
    name = file.filename or "video"
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Use one of: {sorted(ALLOWED_SUFFIX)}",
        )
    content = await file.read()
    if len(content) < 1024:
        raise HTTPException(status_code=400, detail="File too small or empty")
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=300.0, pool=30.0)
    files = {"file": (name, content, file.content_type or "application/octet-stream")}
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=httpx_trust_env()) as client:
            response = await client.post(url, files=files)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": str(e),
                "upstream_url": url,
                "hints": [
                    "Confirm the upstream app is listening (e.g. curl the URL from the same machine/container).",
                    "If this API runs in Docker, localhost points inside the container: set UPSTREAM_PREDICT_BASE or VEGA_PREDICT_URL to the host or compose service hostname.",
                    "If HTTP_PROXY/HTTPS_PROXY is set, internal calls may fail: leave HTTPX_TRUST_ENV unset/false (default) or fix proxy bypass for that host.",
                ],
            },
        ) from e
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text or response.reason_phrase
        raise HTTPException(status_code=response.status_code, detail=detail)
    try:
        return response.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Upstream returned non-JSON body",
        ) from None

