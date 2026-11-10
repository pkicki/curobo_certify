"""Pinned cuRobo TimeOpt adapter status and immutable tensor export helpers.

The native solver's outputs are borrowed buffers. Clone before retaining them.
The exact upstream pin must be installed before constructing a solver adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
from typing import Any

UPSTREAM_SHA = "c301cb00a87012c2808b1a60649c4065c9b3c95a"


class AdapterBlocked(RuntimeError):
    """Raised when pinned upstream/effective-geometry contract is unavailable."""


def require_pinned_upstream() -> None:
    try:
        from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt  # noqa: F401
    except Exception as exc:
        raise AdapterBlocked(f"cuRobo unavailable: {type(exc).__name__}: {exc}") from exc
    try:
        version = importlib.metadata.version("curobo")
    except importlib.metadata.PackageNotFoundError as exc:
        raise AdapterBlocked("cuRobo package metadata unavailable; verify exact git SHA") from exc
    if version != UPSTREAM_SHA:
        raise AdapterBlocked(
            f"installed cuRobo version {version!r} does not attest required git SHA {UPSTREAM_SHA}"
        )


def clone_result(result: Any) -> Any:
    """Clone the upstream borrowed result before any later solver call."""
    if not callable(getattr(result, "clone", None)):
        raise AdapterBlocked("upstream result has no clone() method")
    return result.clone()


def sha256_array(array: Any) -> str:
    """Hash canonical contiguous tensor/array bytes with dtype and shape metadata."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    import numpy as np
    value = np.ascontiguousarray(array)
    header = f"{value.dtype.str}|{','.join(map(str, value.shape))}|".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class GeometryContract:
    status: str = "BLOCKED_ADAPTER"
    reason: str = (
        "Pinned upstream checkout and GPU runtime are unavailable; effective clamped boundary "
        "geometry has not been exported or parity checked."
    )
