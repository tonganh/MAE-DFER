import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mae_dfer.inference import infer_video
from mae_dfer.inference import speech_emotion_infer as speech_emotion

from . import model_registry
from .config import api_save_root, include_resource_stats
from .resource_stats import (
    gpu_peak_mib,
    gpu_reset_peak,
    now_cpu_sec,
    resource_stats_payload,
    vm_rss_kb,
)
from .video_io import resolve_suffix


def predict_bytes_sync(data: bytes, filename: str) -> dict[str, Any]:
    if len(data) < 1024:
        raise ValueError("File too small or empty")
    if not model_registry.model_loaded():
        raise RuntimeError("Model not loaded")

    t0 = time.perf_counter()
    start_ts = datetime.now(timezone.utc).isoformat()
    print(f"[predict] start_ts={start_ts} filename={filename!r} bytes={len(data)}", flush=True)

    suffix = resolve_suffix(filename, data)
    save_root_raw = api_save_root()

    sub: Path | None = None
    cleanup: str | None = None

    if save_root_raw:
        save_root = Path(save_root_raw)
        save_root.mkdir(parents=True, exist_ok=True)
        sub = save_root / uuid.uuid4().hex
        sub.mkdir(parents=True, exist_ok=True)
        path = str(sub / f"video{suffix}")
        with open(path, "wb") as f:
            f.write(data)
    else:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        cleanup = path
        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise

    try:
        want_res = include_resource_stats()
        model, device, inference_args, infer_lock = model_registry.get_model_parts()

        if want_res:
            rss_start = vm_rss_kb()
            cpu_start = now_cpu_sec()
            gpu_reset_peak(device)

        t_video0 = time.perf_counter()
        with infer_lock:
            out = infer_video.predict_video_file(path, model, device, inference_args)
        t_video1 = time.perf_counter()

        out["filename"] = filename or "video"

        if want_res:
            rss_after_video = vm_rss_kb()
            cpu_after_video = now_cpu_sec()
            gpu_video_peak = gpu_peak_mib(device)
            gpu_reset_peak(device)
        else:
            rss_after_video = None
            cpu_after_video = 0.0
            gpu_video_peak = None
            cpu_start = 0.0
            rss_start = None

        t_whisper0 = time.perf_counter()
        try:
            speech_out = speech_emotion.predict_from_video_bytes(
                data,
                filename or "video",
                speech_emotion.SpeechBackend.whisper,
            )
            out["speech_emotion_whisper"] = speech_out.get("whisper")
        except Exception as e:
            out["speech_emotion_whisper_error"] = str(e)
        t_whisper1 = time.perf_counter()

        t1 = time.perf_counter()
        end_ts = datetime.now(timezone.utc).isoformat()
        out["timing"] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "elapsed_sec": round(t1 - t0, 4),
            "video_infer_sec": round(t_video1 - t_video0, 4),
            "whisper_sec": round(t_whisper1 - t_whisper0, 4),
        }

        if want_res:
            rss_end = vm_rss_kb()
            cpu_end = now_cpu_sec()
            gpu_whisper_peak = gpu_peak_mib(device)
            out["resource_stats"] = resource_stats_payload(
                rss_start_kb=rss_start,
                rss_after_video_kb=rss_after_video,
                rss_end_kb=rss_end,
                cpu_start=cpu_start,
                cpu_after_video=cpu_after_video,
                cpu_end=cpu_end,
                gpu_video_peak_mib=gpu_video_peak,
                gpu_whisper_peak_mib=gpu_whisper_peak,
            )

        print(
            "[predict] end_ts="
            + end_ts
            + " elapsed_sec="
            + str(out["timing"]["elapsed_sec"])
            + " video_infer_sec="
            + str(out["timing"]["video_infer_sec"])
            + " whisper_sec="
            + str(out["timing"]["whisper_sec"]),
            flush=True,
        )

        if sub is not None:
            out["saved_dir"] = str(sub.resolve())
            out["saved_video"] = str(Path(path).resolve())
            out_path = sub / "result.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            out["saved_output"] = str(out_path.resolve())

        return out
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass

