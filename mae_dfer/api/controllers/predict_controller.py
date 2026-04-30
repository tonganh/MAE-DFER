import asyncio
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from mae_dfer.inference import speech_emotion_infer as speech_emotion

from mae_dfer.api.services import config, inference_service, model_registry, upstream_service
from mae_dfer.api.services.video_io import ALLOWED_SUFFIX


router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_registry.model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    name = file.filename or "video"
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Use one of: {sorted(ALLOWED_SUFFIX)}",
        )
    content = await file.read()
    try:
        return await asyncio.to_thread(inference_service.predict_bytes_sync, content, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/predict-vega")
async def predict_vega(file: UploadFile = File(...)):
    return await upstream_service.forward_video_multipart_to_url(config.vega_predict_url(), file)


@router.post("/predict-speech-emotion")
async def predict_speech_emotion(file: UploadFile = File(...)):
    return await upstream_service.forward_video_multipart_to_url(
        config.speech_emotion_predict_url(), file
    )


@router.post("/predict/video-speech-emotion")
async def predict_video_speech_emotion(
    file: UploadFile = File(...),
    backend: speech_emotion.SpeechBackend = speech_emotion.SpeechBackend.whisper,
):
    name = file.filename or "video"
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Use one of: {sorted(ALLOWED_SUFFIX)}",
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

