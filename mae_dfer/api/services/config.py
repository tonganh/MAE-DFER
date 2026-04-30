import os


DEFAULT_CHECKPOINT = "./saved/model/finetuning/dfew/dfrew_fold05/checkpoint_dfew_fold5.pth"


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


def include_resource_stats() -> bool:
    return env_truthy("API_INCLUDE_RESOURCE_STATS")


def stream_segment_sec() -> float:
    try:
        return float(os.environ.get("STREAM_SEGMENT_SEC", "10"))
    except ValueError:
        return 10.0

