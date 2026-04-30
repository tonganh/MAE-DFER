import os
import re
from pathlib import Path

import httpx


_DFEW_FOLD5_FILE_ID = "1wmXO4M2kjpAOnvof8CmpJE6wUrxMUOgw"


def _google_drive_uc_url(file_id: str, confirm: str | None = None) -> str:
    base = f"https://drive.google.com/uc?id={file_id}&export=download"
    if confirm:
        return f"{base}&confirm={confirm}"
    return base


def _extract_confirm_token(html: str) -> str | None:
    m = re.search(r"confirm=([0-9A-Za-z_]+)", html)
    if m:
        return m.group(1)
    return None


def download_dfew_fold5_checkpoint(dst_path: str) -> None:
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=600.0)) as client:
        r1 = client.get(_google_drive_uc_url(_DFEW_FOLD5_FILE_ID))
        confirm = None
        content_type = r1.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            confirm = _extract_confirm_token(r1.text)
        if confirm:
            r = client.get(_google_drive_uc_url(_DFEW_FOLD5_FILE_ID, confirm=confirm))
        else:
            r = r1
        r.raise_for_status()

        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)

    os.replace(tmp, dst)


def ensure_default_checkpoint_exists(default_checkpoint_path: str) -> None:
    p = Path(default_checkpoint_path)
    if p.is_file():
        return
    download_dfew_fold5_checkpoint(str(p))

