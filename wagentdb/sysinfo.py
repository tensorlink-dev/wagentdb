"""Best-effort capture of environment + git info for reproducibility.

Everything here degrades gracefully: if torch/transformers aren't installed, or
the cwd isn't a git repo, the relevant keys are simply omitted. This keeps
wagentdb framework-agnostic — it captures whatever stack the experiment happens
to use without depending on any of it.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from typing import Any, Dict, Optional


def collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }

    # PyTorch + accelerator details (the common case for LLM / DL training).
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["mps_available"] = True
    except Exception:
        pass

    # Versions of common training libraries, if present.
    for lib in ("transformers", "datasets", "accelerate", "peft", "trl",
                "lightning", "pytorch_lightning", "jax", "numpy"):
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", None)
            if ver:
                info[lib] = ver
        except Exception:
            pass

    return info


def collect_git_info(path: str = ".") -> Dict[str, Optional[str]]:
    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.check_output(
                ["git", "-C", path, *args],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return out.decode().strip()
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    if not commit:
        return {}
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "remote": _git("config", "--get", "remote.origin.url"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "true" if status else "false",
    }
