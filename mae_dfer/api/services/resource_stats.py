import time
from typing import Any

import torch


def vm_rss_kb() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def kb_to_mib(kb: int | None) -> float | None:
    if kb is None:
        return None
    return round(kb / 1024, 2)


def gpu_reset_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_mib(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    return round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)


def now_cpu_sec() -> float:
    return time.process_time()


def resource_stats_payload(
    *,
    rss_start_kb: int | None,
    rss_after_video_kb: int | None,
    rss_end_kb: int | None,
    cpu_start: float,
    cpu_after_video: float,
    cpu_end: float,
    gpu_video_peak_mib: float | None,
    gpu_whisper_peak_mib: float | None,
) -> dict[str, Any]:
    return {
        "rss_resident_set_mib": {
            "before_video_infer": kb_to_mib(rss_start_kb),
            "after_video_infer": kb_to_mib(rss_after_video_kb),
            "after_request": kb_to_mib(rss_end_kb),
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

