import os
import threading

import torch

from mae_dfer.inference import infer_video

from .checkpoint_service import ensure_default_checkpoint_exists
from .config import DEFAULT_CHECKPOINT, checkpoint_path


_inference_args = None
_model = None
_device: torch.device | None = None
_infer_lock = threading.Lock()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def load_model() -> None:
    global _inference_args, _model, _device
    checkpoint = checkpoint_path()
    if checkpoint == DEFAULT_CHECKPOINT and "CHECKPOINT_PATH" not in os.environ:
        ensure_default_checkpoint_exists(checkpoint)
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Set CHECKPOINT_PATH to a valid .pth file."
        )
    device_str = os.environ.get("DEVICE", "cuda")
    _device = resolve_device(device_str)
    _inference_args = infer_video.default_inference_args()
    _model = infer_video.build_model(_inference_args)
    infer_video.load_checkpoint(_model, checkpoint)
    _model.eval()
    _model.to(_device)


def model_loaded() -> bool:
    return _model is not None and _inference_args is not None and _device is not None


def get_model_parts():
    return _model, _device, _inference_args, _infer_lock


def device_str() -> str | None:
    return str(_device) if _device is not None else None

