import subprocess
import tempfile
import threading
import uuid
from enum import Enum
from pathlib import Path

import numpy as np
import torch

_ALLOWED_SUFFIX = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

_wav2vec_lock = threading.Lock()
_whisper_lock = threading.Lock()
_wav2vec_pipe = None
_whisper_model = None
_whisper_extractor = None
_whisper_id2label = None


class SpeechBackend(str, Enum):
    wav2vec2 = "wav2vec2"
    whisper = "whisper"
    both = "both"


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


def extract_audio_wav(video_path: str, wav_path: str, sample_rate: int = 16000) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        wav_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip() or f"ffmpeg exited {p.returncode}"
        raise RuntimeError(msg)


def _get_wav2vec_pipe():
    global _wav2vec_pipe
    with _wav2vec_lock:
        if _wav2vec_pipe is None:
            from transformers import pipeline

            dev = 0 if torch.cuda.is_available() else -1
            _wav2vec_pipe = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=dev,
            )
    return _wav2vec_pipe


def _wav2vec_scores(wav_path: str) -> dict:
    pipe = _get_wav2vec_pipe()
    with _wav2vec_lock:
        raw = pipe(wav_path, top_k=50)
    if not raw:
        return {"label": None, "score": None, "scores": []}
    rows = [{"label": x["label"], "score": float(x["score"])} for x in raw]
    top = rows[0]
    return {"label": top["label"], "score": top["score"], "scores": rows}


def _whisper_scores(wav_path: str, max_duration: float = 30.0) -> dict:
    import librosa
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    global _whisper_model, _whisper_extractor, _whisper_id2label
    with _whisper_lock:
        if _whisper_model is None:
            model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
            _whisper_model = AutoModelForAudioClassification.from_pretrained(model_id)
            _whisper_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
            _whisper_id2label = _whisper_model.config.id2label
            if torch.cuda.is_available():
                _whisper_model = _whisper_model.cuda()
            _whisper_model.eval()
        model = _whisper_model
        feature_extractor = _whisper_extractor
        id2label = _whisper_id2label
    audio_array, sr = librosa.load(wav_path, sr=None)
    target_sr = feature_extractor.sampling_rate
    if sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
    max_length = int(target_sr * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
    inputs = feature_extractor(
        audio_array,
        sampling_rate=target_sr,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    with _whisper_lock:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        n = probs.shape[0]

        def _lbl(i: int) -> str:
            m = id2label
            if isinstance(m, dict):
                v = m.get(i)
                if v is None:
                    v = m.get(str(i))
                return str(v if v is not None else "")
            return str(m[i])

        rows = sorted(
            [{"label": _lbl(i), "score": float(probs[i].item())} for i in range(n)],
            key=lambda x: x["score"],
            reverse=True,
        )
        return {"label": _lbl(pred_id), "score": float(probs[pred_id].item()), "scores": rows}


def predict_from_video_bytes(data: bytes, filename: str, backend: SpeechBackend) -> dict:
    if len(data) < 1024:
        raise ValueError("File too small or empty")
    suffix = _resolve_suffix(filename, data)
    if suffix not in _ALLOWED_SUFFIX:
        raise ValueError(f"Unsupported file type {suffix!r}")
    out: dict = {
        "filename": filename or "video",
        "audio_sample_rate_hz": 16000,
        "backend": backend.value,
    }
    tmp_root = Path(tempfile.gettempdir()) / "mae_dfer_speech" / uuid.uuid4().hex
    tmp_root.mkdir(parents=True, exist_ok=True)
    video_path = str(tmp_root / f"input{suffix}")
    wav_path = str(tmp_root / "audio.wav")
    try:
        with open(video_path, "wb") as f:
            f.write(data)
        extract_audio_wav(video_path, wav_path, sample_rate=16000)
        if backend in (SpeechBackend.wav2vec2, SpeechBackend.both):
            out["wav2vec2"] = _wav2vec_scores(wav_path)
        if backend in (SpeechBackend.whisper, SpeechBackend.both):
            out["whisper"] = _whisper_scores(wav_path)
    finally:
        try:
            for p in tmp_root.iterdir():
                p.unlink(missing_ok=True)
            tmp_root.rmdir()
        except OSError:
            pass
    return out
