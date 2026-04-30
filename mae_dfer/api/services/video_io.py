from pathlib import Path


ALLOWED_SUFFIX = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def suffix_from_magic(data: bytes) -> str:
    if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".mp4"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return ".avi"
    return ".mp4"


def resolve_suffix(filename: str, data: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_SUFFIX:
        return suffix
    return suffix_from_magic(data)

