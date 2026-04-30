from fastapi import APIRouter

from mae_dfer.api.services import config, model_registry


router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "device": model_registry.device_str(),
        "checkpoint": config.checkpoint_path(),
        "vega_predict_url": config.vega_predict_url(),
        "speech_emotion_predict_url": config.speech_emotion_predict_url(),
        "video_speech_emotion_path": "/predict/video-speech-emotion",
    }

