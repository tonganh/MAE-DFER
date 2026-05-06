import subprocess
from pathlib import Path


def probe_duration_sec(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip() or f"ffprobe exited {p.returncode}"
        raise RuntimeError(msg)
    s = (p.stdout or "").strip()
    if not s or s == "N/A":
        raise RuntimeError("Could not read video duration")
    return float(s)


def extract_segment_mp4(input_path: str, start_sec: float, duration_sec: float, output_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_sec),
        "-i",
        input_path,
        "-t",
        str(duration_sec),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip() or f"ffmpeg exited {p.returncode}"
        raise RuntimeError(msg)
    if not Path(output_path).is_file() or Path(output_path).stat().st_size < 1024:
        raise RuntimeError("Segment output missing or too small")


def iter_chunk_windows(total_sec: float, chunk_sec: float) -> list[tuple[float, float]]:
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < total_sec - 1e-6:
        end = min(start + chunk_sec, total_sec)
        out.append((start, end))
        start = end
    return out
