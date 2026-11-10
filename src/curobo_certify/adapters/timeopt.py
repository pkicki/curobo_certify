"""Pinned cuRobo TimeOpt adapter status and immutable tensor export helpers.

The native solver's outputs are borrowed buffers. Clone before retaining them.
The exact upstream pin must be installed before constructing a solver adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any

UPSTREAM_SHA = "c301cb00a87012c2808b1a60649c4065c9b3c95a"


class AdapterBlocked(RuntimeError):
    """Raised when pinned upstream/effective-geometry contract is unavailable."""


def require_pinned_upstream() -> None:
    try:
        import curobo
        from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt  # noqa: F401
    except Exception as exc:
        raise AdapterBlocked(f"cuRobo unavailable: {type(exc).__name__}: {exc}") from exc

    # The distribution is named ``nvidia-curobo`` and its PEP 440 version is
    # deliberately not a Git SHA. Verify the source tree that Python imported.
    package_dir = Path(curobo.__file__).resolve().parent
    repo = next((parent for parent in package_dir.parents if (parent / ".git").exists()), None)
    if repo is None:
        raise AdapterBlocked(f"imported cuRobo source {package_dir} is not inside a Git worktree")
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdapterBlocked(f"cannot verify imported cuRobo Git revision: {exc}") from exc
    if revision != UPSTREAM_SHA:
        raise AdapterBlocked(
            f"imported cuRobo source revision {revision!r} does not match {UPSTREAM_SHA}"
        )


def clone_result(result: Any) -> Any:
    """Clone the upstream borrowed result before any later solver call."""
    if not callable(getattr(result, "clone", None)):
        raise AdapterBlocked("upstream result has no clone() method")
    return result.clone()


def clone_effective_trajectory(timeopt: Any) -> tuple[Any, Any]:
    """Retain the native result and its post-boundary effective spline geometry.

    The upstream method builds the effective complete support from the start
    boundary, action controls, and goal support. Both returned objects are
    detached from reusable solver buffers before another call can mutate them.
    """
    result = clone_result(timeopt._result)
    try:
        trajectory = timeopt._spline_trajectory().clone()
    except Exception as exc:
        raise AdapterBlocked(f"effective geometry export failed: {type(exc).__name__}: {exc}") from exc
    return result, trajectory


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
    reason: str = "Run the pinned native parity export and retain its hashes before certification."
