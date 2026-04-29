import asyncio
import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import infer_video
import speech_emotion_infer as speech_emotion

_DEFAULT_CHECKPOINT = ("./saved/model/finetuning/dfew/dfrew_fold05/checkpoint_dfew_fold5.pth")
_UPSTREAM_BASE = os.environ.get("UPSTREAM_PREDICT_BASE", "http://localhost:8001")
_VEGA_PREDICT_URL = os.environ.get(
    "VEGA_PREDICT_URL", f"{_UPSTREAM_BASE.rstrip('/')}/predict"
)
_SPEECH_EMOTION_PREDICT_URL = os.environ.get(
    "SPEECH_EMOTION_PREDICT_URL",
    f"{_UPSTREAM_BASE.rstrip('/')}/predict/speech_emotion",
)

_inference_args = None
_model = None
_device = None
_infer_lock = threading.Lock()


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes")


def _vm_rss_kb():
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None


def _kb_to_mib(kb):
    if kb is None:
        return None
    return round(kb / 1024, 2)


def _gpu_reset_peak(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _gpu_peak_mib(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    return round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)


def _httpx_trust_env() -> bool:
    v = os.environ.get("HTTPX_TRUST_ENV", "").strip().lower()
    return v in ("1", "true", "yes")


def _resolve_device(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _load_model():
    global _inference_args, _model, _device
    checkpoint = os.environ.get("CHECKPOINT_PATH", _DEFAULT_CHECKPOINT)
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Set CHECKPOINT_PATH to a valid .pth file."
        )
    device_str = os.environ.get("DEVICE", "cuda")
    _device = _resolve_device(device_str)
    _inference_args = infer_video.default_inference_args()
    _model = infer_video.build_model(_inference_args)
    infer_video.load_checkpoint(_model, checkpoint)
    _model.eval()
    _model.to(_device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="MAE-DFER emotion inference", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(_device) if _device else None,
        "checkpoint": os.environ.get("CHECKPOINT_PATH", _DEFAULT_CHECKPOINT),
        "vega_predict_url": _VEGA_PREDICT_URL,
        "speech_emotion_predict_url": _SPEECH_EMOTION_PREDICT_URL,
        "video_speech_emotion_path": "/predict/video-speech-emotion",
    }


_ALLOWED_SUFFIX = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def _suffix_from_magic(data: bytes) -> str:
    if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".mp4"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return ".avi"
    return ".mp4"


def _resolve_suffix(filename: str, data: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in _ALLOWED_SUFFIX:
        return suffix
    return _suffix_from_magic(data)


def _predict_bytes_sync(data: bytes, filename: str) -> dict:
    if len(data) < 1024:
        raise ValueError("File too small or empty")
    t0 = time.perf_counter()
    start_ts = datetime.now(timezone.utc).isoformat()
    print(f"[predict] start_ts={start_ts} filename={filename!r} bytes={len(data)}", flush=True)
    suffix = _resolve_suffix(filename, data)
    if "API_SAVE_DIR" not in os.environ:
        save_root_raw = "api_saved"
    else:
        save_root_raw = os.environ["API_SAVE_DIR"].strip()
    sub = None
    cleanup = None
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
        want_res = _env_truthy("API_INCLUDE_RESOURCE_STATS")
        if want_res:
            rss_start = _vm_rss_kb()
            cpu_start = time.process_time()
            _gpu_reset_peak(_device)
        t_video0 = time.perf_counter()
        with _infer_lock:
            out = infer_video.predict_video_file(path, _model, _device, _inference_args)
        t_video1 = time.perf_counter()
        out["filename"] = filename or "video"
        if want_res:
            rss_after_video = _vm_rss_kb()
            cpu_after_video = time.process_time()
            gpu_video_peak_mib = _gpu_peak_mib(_device)
            _gpu_reset_peak(_device)
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
            rss_end = _vm_rss_kb()
            cpu_end = time.process_time()
            gpu_whisper_peak_mib = _gpu_peak_mib(_device)
            out["resource_stats"] = {
                "rss_resident_set_mib": {
                    "before_video_infer": _kb_to_mib(rss_start),
                    "after_video_infer": _kb_to_mib(rss_after_video),
                    "after_request": _kb_to_mib(rss_end),
                },
                "process_cpu_sec": {
                    "video_infer": round(cpu_after_video - cpu_start, 4),
                    "speech_whisper": round(cpu_end - cpu_after_video, 4),
                    "video_plus_whisper": round(cpu_end - cpu_start, 4),
                },
                "gpu_cuda_peak_reserved_mib": {
                    "video_infer": gpu_video_peak_mib,
                    "speech_whisper": gpu_whisper_peak_mib,
                },
            }
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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None or _inference_args is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    name = file.filename or "video"
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in _ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Use one of: {sorted(_ALLOWED_SUFFIX)}",
        )
    content = await file.read()
    try:
        return await asyncio.to_thread(_predict_bytes_sync, content, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


async def _forward_video_multipart_to_url(url: str, file: UploadFile) -> dict:
    name = file.filename or "video"
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in _ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Use one of: {sorted(_ALLOWED_SUFFIX)}",
        )
    content = await file.read()
    if len(content) < 1024:
        raise HTTPException(status_code=400, detail="File too small or empty")
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=300.0, pool=30.0)
    files = {"file": (name, content, file.content_type or "application/octet-stream")}
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=_httpx_trust_env()) as client:
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


@app.post("/predict-vega")
async def predict_vega(file: UploadFile = File(...)):
    return await _forward_video_multipart_to_url(_VEGA_PREDICT_URL, file)


@app.post("/predict-speech-emotion")
async def predict_speech_emotion(file: UploadFile = File(...)):
    return await _forward_video_multipart_to_url(_SPEECH_EMOTION_PREDICT_URL, file)


@app.post("/predict/video-speech-emotion")
async def predict_video_speech_emotion(
    file: UploadFile = File(...),
    backend: speech_emotion.SpeechBackend = speech_emotion.SpeechBackend.whisper,
):
    name = file.filename or "video"
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in _ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Use one of: {sorted(_ALLOWED_SUFFIX)}",
        )
    content = await file.read()
    try:
        return await asyncio.to_thread(
            speech_emotion.predict_from_video_bytes,
            content,
            name,
            backend,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "audio_extraction_failed", "message": str(e)},
        ) from e


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    if _model is None or _inference_args is None:
        await websocket.close(code=1011)
        return
    segment_index = 0
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"error": "invalid_json", "segment_index": segment_index}
                    )
                    continue
                if payload.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                continue
            if msg.get("bytes") is not None:
                data = msg["bytes"]
                name = f"segment_{segment_index}.bin"
                try:
                    out = await asyncio.to_thread(_predict_bytes_sync, data, name)
                    out["segment_index"] = segment_index
                    out["segment_duration_hint_sec"] = float(
                        os.environ.get("STREAM_SEGMENT_SEC", "10")
                    )
                    await websocket.send_json(out)
                    segment_index += 1
                except ValueError as e:
                    await websocket.send_json(
                        {
                            "error": str(e),
                            "segment_index": segment_index,
                        }
                    )
    except WebSocketDisconnect:
        pass


def main():
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    ws_max = int(os.environ.get("WS_MAX_SIZE", str(50 * 1024 * 1024)))
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=1,
        ws_max_size=ws_max,
    )


if __name__ == "__main__":
    main()
