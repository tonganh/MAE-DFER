import os
from pathlib import Path


# DEFAULT_CHECKPOINT = "./saved/model/finetuning/dfew/dfrew_fold05/checkpoint_dfew_fold5.pth"
DEFAULT_CHECKPOINT = "./saved/model/readme_checkpoints/finetuning/ferv39k/checkpoint.pth"

def env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes")


def httpx_trust_env() -> bool:
    v = os.environ.get("HTTPX_TRUST_ENV", "").strip().lower()
    return v in ("1", "true", "yes")


def checkpoint_path() -> str:
    return os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT)


def upstream_base() -> str:
    return os.environ.get("UPSTREAM_PREDICT_BASE", "http://localhost:8001")


def vega_predict_url() -> str:
    base = upstream_base().rstrip("/")
    return os.environ.get("VEGA_PREDICT_URL", f"{base}/predict")


def speech_emotion_predict_url() -> str:
    base = upstream_base().rstrip("/")
    return os.environ.get("SPEECH_EMOTION_PREDICT_URL", f"{base}/predict/speech_emotion")


def api_save_root() -> str:
    if "API_SAVE_DIR" in os.environ:
        return os.environ["API_SAVE_DIR"].strip()
    return os.environ.get("API_SAVE_ROOT", "data/api_saved")


def chunk_prediction_save_root() -> Path:
    alt = os.environ.get("API_CHUNK_SAVE_ROOT", "").strip()
    if alt:
        return Path(alt).expanduser()
    return Path.cwd() / "data" / "chunk_videos"


def include_resource_stats() -> bool:
    return env_truthy("API_INCLUDE_RESOURCE_STATS")


def include_transcript() -> bool:
    if "API_INCLUDE_TRANSCRIPT" not in os.environ:
        return True
    return env_truthy("API_INCLUDE_TRANSCRIPT")


def openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()


def openai_eval_model() -> str:
    m = os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-mini").strip()
    return m or "gpt-4o-mini"


def llm_emotion_audio_weight() -> float:
    try:
        w = float(os.environ.get("API_LLM_EMOTION_AUDIO_WEIGHT", "0.65"))
    except ValueError:
        return 0.65
    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return w


def stream_segment_sec() -> float:
    try:
        return float(os.environ.get("STREAM_SEGMENT_SEC", "10"))
    except ValueError:
        return 10.0

