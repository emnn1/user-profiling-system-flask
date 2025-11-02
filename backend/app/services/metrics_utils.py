"""Utility helpers for runtime metrics and resource monitoring."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import torch


_PROCESS = psutil.Process(os.getpid())


def _bytes_to_mb(value: int) -> float:
    """Convert bytes to megabytes with one decimal precision."""
    return round(value / (1024 ** 2), 2)


def capture_resource_snapshot() -> Dict[str, Any]:
    """Capture a lightweight snapshot of CPU, memory and GPU usage."""

    try:
        cpu_percent = _PROCESS.cpu_percent(interval=None)
    except Exception:
        cpu_percent = None

    try:
        memory_info = _PROCESS.memory_info()
        rss_mb = _bytes_to_mb(memory_info.rss)
        vms_mb = _bytes_to_mb(memory_info.vms)
    except Exception:
        rss_mb = None
        vms_mb = None

    gpu_info: Optional[Dict[str, Any]] = None
    if torch.cuda.is_available():
        try:
            device_index = torch.cuda.current_device()
            gpu_info = {
                "device": torch.cuda.get_device_name(device_index),
                "memory_allocated_mb": _bytes_to_mb(int(torch.cuda.memory_allocated(device_index))),
                "memory_reserved_mb": _bytes_to_mb(int(torch.cuda.memory_reserved(device_index))),
            }
        except Exception:
            gpu_info = {"device": "unknown", "memory_allocated_mb": None, "memory_reserved_mb": None}

    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "cpu_percent": cpu_percent,
        "memory_rss_mb": rss_mb,
        "memory_vms_mb": vms_mb,
        "gpu": gpu_info,
        "pid": _PROCESS.pid,
    }


__all__ = ["capture_resource_snapshot"]
