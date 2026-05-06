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
from .config import (
    api_save_root,
    chunk_prediction_save_root,
    env_truthy,
    include_resource_stats,
    include_transcript,
    openai_api_key,
)
from .resource_stats import (
    gpu_peak_mib,
    gpu_reset_peak,
    now_cpu_sec,
    resource_stats_payload,
    vm_rss_kb,
)
from .video_chunking import extract_segment_mp4, iter_chunk_windows, probe_duration_sec
from .video_io import resolve_suffix


def _infer_video_only(video_path: str) -> dict[str, Any]:
    model, device, inference_args, infer_lock = model_registry.get_model_parts()
    with infer_lock:
        return infer_video.predict_video_file(
            video_path,
            model,
            device,
            inference_args,
            with_transcript=include_transcript(),
        )


def _attach_whisper_to_out(out: dict[str, Any], speech_bytes: bytes, filename: str) -> None:
    try:
        speech_out = speech_emotion.predict_from_video_bytes(
            speech_bytes,
            filename or "video",
            speech_emotion.SpeechBackend.whisper,
        )
        out["speech_emotion_whisper"] = speech_out.get("whisper")
    except Exception as e:
        out["speech_emotion_whisper_error"] = str(e)


def _attach_llm_emotion_evaluation(out: dict[str, Any]) -> None:
    if "API_LLM_EMOTION_EVAL" in os.environ and not env_truthy("API_LLM_EMOTION_EVAL"):
        out["llm_emotion_eval"] = {"skipped": True, "reason": "API_LLM_EMOTION_EVAL disabled"}
        return
    key = openai_api_key()
    if not key:
        out["llm_emotion_eval"] = {"skipped": True, "reason": "OPENAI_API_KEY not set"}
        return
    try:
        from . import llm_emotion_eval

        out["llm_emotion_eval"] = llm_emotion_eval.run_llm_emotion_evaluation(out, api_key=key)
    except Exception as e:
        out["llm_emotion_eval"] = {"skipped": False, "error": str(e)}


def _promote_llm_fusion_fields(out: dict[str, Any]) -> None:
    le = out.get("llm_emotion_eval")
    if not isinstance(le, dict) or le.get("skipped"):
        return
    fa = le.get("final_agent")
    if not isinstance(fa, dict):
        return
    fe = fa.get("final_emotion")
    if isinstance(fe, str) and fe.strip():
        out["fusion_final_emotion"] = fe.strip()
    if fa.get("confidence") is not None:
        out["fusion_final_confidence"] = fa["confidence"]


def _run_video_speech_predict(
    video_path: str, speech_bytes: bytes, filename: str
) -> tuple[dict[str, Any], float, float]:
    t_video0 = time.perf_counter()
    out = _infer_video_only(video_path)
    t_video1 = time.perf_counter()
    out["filename"] = filename or "video"
    t_whisper0 = time.perf_counter()
    _attach_whisper_to_out(out, speech_bytes, filename)
    t_whisper1 = time.perf_counter()
    _attach_llm_emotion_evaluation(out)
    _promote_llm_fusion_fields(out)
    if isinstance(out.get("llm_emotion_eval"), dict) and "elapsed_sec" in out["llm_emotion_eval"]:
        out["timing_llm_emotion_eval_sec"] = out["llm_emotion_eval"]["elapsed_sec"]
    return out, t_video1 - t_video0, t_whisper1 - t_whisper0


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
        model, device, _, _ = model_registry.get_model_parts()

        if want_res:
            rss_start = vm_rss_kb()
            cpu_start = now_cpu_sec()
            gpu_reset_peak(device)

        t_video0 = time.perf_counter()
        out = _infer_video_only(path)
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
        _attach_whisper_to_out(out, data, filename or "video")
        t_whisper1 = time.perf_counter()
        _attach_llm_emotion_evaluation(out)
        _promote_llm_fusion_fields(out)

        t1 = time.perf_counter()
        end_ts = datetime.now(timezone.utc).isoformat()
        timing: dict[str, Any] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "elapsed_sec": round(t1 - t0, 4),
            "video_infer_sec": round(t_video1 - t_video0, 4),
            "whisper_sec": round(t_whisper1 - t_whisper0, 4),
        }
        le = out.get("llm_emotion_eval")
        if isinstance(le, dict) and "elapsed_sec" in le:
            timing["llm_emotion_eval_sec"] = le["elapsed_sec"]
        out["timing"] = timing

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


def predict_bytes_chunked_sync(
    data: bytes, filename: str, chunk_seconds: float = 1.5
) -> dict[str, Any]:
    if len(data) < 1024:
        raise ValueError("File too small or empty")
    if not model_registry.model_loaded():
        raise RuntimeError("Model not loaded")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    t0 = time.perf_counter()
    start_ts = datetime.now(timezone.utc).isoformat()
    suffix = resolve_suffix(filename, data)
    save_root = chunk_prediction_save_root()
    save_root.mkdir(parents=True, exist_ok=True)
    sub = save_root / uuid.uuid4().hex
    sub.mkdir(parents=True, exist_ok=True)
    chunk_dir = sub / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    full_path = str(sub / f"video{suffix}")
    with open(full_path, "wb") as f:
        f.write(data)

    duration = probe_duration_sec(full_path)
    windows = iter_chunk_windows(duration, chunk_seconds)
    base_stem = Path(filename or "video").name
    chunks: list[dict[str, Any]] = []
    sum_video = 0.0
    sum_whisper = 0.0
    for i, (start_sec, end_sec) in enumerate(windows):
        seg_dur = end_sec - start_sec
        chunk_path = chunk_dir / f"chunk_{i:04d}.mp4"
        extract_segment_mp4(full_path, start_sec, seg_dur, str(chunk_path))
        with open(chunk_path, "rb") as f:
            chunk_bytes = f.read()
        if len(chunk_bytes) < 1024:
            raise ValueError(f"Chunk {i} too small after extraction")
        label = f"{base_stem}#chunk{i}"
        o, vi, wi = _run_video_speech_predict(str(chunk_path), chunk_bytes, label)
        o["chunk_index"] = i
        o["start_sec"] = round(start_sec, 4)
        o["end_sec"] = round(end_sec, 4)
        o["duration_sec"] = round(seg_dur, 4)
        o["chunk_video_path"] = str(chunk_path.resolve())
        o["timing"] = {
            "video_infer_sec": round(vi, 4),
            "whisper_sec": round(wi, 4),
        }
        lm = o.pop("timing_llm_emotion_eval_sec", None)
        if lm is not None:
            o["timing"]["llm_emotion_eval_sec"] = lm
        chunks.append(o)
        sum_video += vi
        sum_whisper += wi

    t1 = time.perf_counter()
    end_ts = datetime.now(timezone.utc).isoformat()
    agg: dict[str, Any] = {
        "filename": filename or "video",
        "chunk_seconds": chunk_seconds,
        "video_duration_sec": round(duration, 4),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "timing": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "elapsed_sec": round(t1 - t0, 4),
            "sum_video_infer_sec": round(sum_video, 4),
            "sum_whisper_sec": round(sum_whisper, 4),
        },
    }
    agg["saved_dir"] = str(sub.resolve())
    agg["saved_video"] = str(Path(full_path).resolve())
    agg["chunk_videos_dir"] = str(chunk_dir.resolve())
    out_path = sub / "result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    agg["saved_output"] = str(out_path.resolve())
    return agg

