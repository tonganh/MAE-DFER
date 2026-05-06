import asyncio
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from mae_dfer.inference import speech_emotion_infer as speech_emotion

from mae_dfer.api.services import config, inference_service, model_registry, upstream_service
from mae_dfer.api.services.video_io import ALLOWED_SUFFIX


router = APIRouter()


@router.post(
    "/predict",
    summary="Video + speech emotion + LLM fusion",
    description=(
        "Runs the vision model (`predicted_label`, probabilities), Whisper-based **audio** emotion, "
        "then an OpenAI **fusion** step that sees **only each modality's top label** (no per-class scores in the LLM prompt; "
        "relative weighting via `API_LLM_EMOTION_AUDIO_WEIGHT`). When the key is present and `API_LLM_EMOTION_EVAL` is not disabled, "
        "the response includes `llm_emotion_eval` and top-level `fusion_final_emotion` (LLM returns only that label). "
        "`chunked=true` runs the same pipeline for each temporal chunk (each chunk has its own fusion fields)."
    ),
)
async def predict(
    file: UploadFile = File(...),
    chunked: bool = Query(False),
    chunk_seconds: float = Query(1.5, gt=0),
):
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
        if chunked:
            return await asyncio.to_thread(
                inference_service.predict_bytes_chunked_sync, content, name, chunk_seconds
            )
        return await asyncio.to_thread(inference_service.predict_bytes_sync, content, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


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

