import os
from pathlib import Path

import gdown


_DFEW_FOLD5_FILE_ID = "1wmXO4M2kjpAOnvof8CmpJE6wUrxMUOgw"


def _reject_if_html(path: Path, file_id: str) -> None:
    buf = path.read_bytes()[:512].lstrip()
    if buf.startswith(b"<!") or buf.lower().startswith(b"<html"):
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"gdown saved HTML instead of weights (id={file_id}); try upgrading gdown or use a browser export."
        )


def download_google_drive_file(file_id: str, dst_path: str) -> None:
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    out = gdown.download(
        id=file_id,
        output=str(tmp),
        quiet=False,
        resume=True,
        fuzzy=False,
    )
    if out is None or not tmp.is_file() or tmp.stat().st_size == 0:
        raise RuntimeError(f"gdown failed for id={file_id}")
    _reject_if_html(tmp, file_id)
    os.replace(tmp, dst)


def download_dfew_fold5_checkpoint(dst_path: str) -> None:
    download_google_drive_file(_DFEW_FOLD5_FILE_ID, dst_path)


def ensure_default_checkpoint_exists(default_checkpoint_path: str) -> None:
    p = Path(default_checkpoint_path)
    if p.is_file():
        return
    download_dfew_fold5_checkpoint(str(p))
