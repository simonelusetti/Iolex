"""Small runtime helpers shared by the NER data and tagger code."""
from pathlib import Path
from typing import Any

import torch


LAUNCH_CWD = Path.cwd()


def to_absolute_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (LAUNCH_CWD / path).resolve()


_interop_threads_configured = False


def configure_runtime(runtime_cfg: dict) -> tuple[dict, bool]:
    """Apply PyTorch runtime settings and fall back when CUDA is unavailable."""
    global _interop_threads_configured
    if runtime_cfg.get("threads") is not None:
        torch.set_num_threads(int(runtime_cfg["threads"]))
    if not _interop_threads_configured and runtime_cfg.get("interop_threads") is not None:
        torch.set_num_interop_threads(int(runtime_cfg["interop_threads"]))
        _interop_threads_configured = True

    tf32 = bool(runtime_cfg.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    requested = str(runtime_cfg.get("device", "cpu"))
    fell_back = requested.startswith("cuda") and not torch.cuda.is_available()
    runtime_cfg["device"] = requested if not fell_back else "cpu"
    return runtime_cfg, fell_back


def to_device(device: str | torch.device, batch: dict) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

