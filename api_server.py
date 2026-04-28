import asyncio
import json
import os
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
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
        with _infer_lock:
            out = infer_video.predict_video_file(path, _model, _device, _inference_args)
        out["filename"] = filename or "video"
        try:
            speech_out = speech_emotion.predict_from_video_bytes(
                data,
                filename or "video",
                speech_emotion.SpeechBackend.whisper,
            )
            out["speech_emotion_whisper"] = speech_out.get("whisper")
        except Exception as e:
            out["speech_emotion_whisper_error"] = str(e)
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
