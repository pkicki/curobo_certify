"""CPU-only exploratory hook for a retained native pilot NPZ.

The timer includes NPZ loading, exact spline extraction, limit/contract
construction, outward K3 certification, and explicit proof replay.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np

from .boundary_reference import Boundaries
from .piecewise_cpu import certify_piecewise_k3, replay_piecewise_k3
from .spline_polynomials import extract_timeopt_spans


def certify_and_replay_npz(blob: bytes) -> dict:
    with np.load(BytesIO(blob), allow_pickle=False) as arrays:
        spans = extract_timeopt_spans(arrays["effective_geometry"].copy(), arrays["timing_controls"].copy())
        q_start = arrays["requested_start"][0].astype(np.float64).tolist()
        q_end = arrays["requested_goal"][0].astype(np.float64).tolist()
        limits = [np.minimum(np.abs(arrays[name][0]), np.abs(arrays[name][1])).astype(float).tolist()
                  for name in ("velocity_limits", "acceleration_limits", "jerk_limits")]
        targets = Boundaries.from_values(q_start=q_start, q_end=q_end,
                                         x_start=spans[0].at("x", 0)[0],
                                         x_end=spans[-1].at("x", 1)[0], strict_rest=True)
        result = certify_piecewise_k3(spans, targets, *limits, max_depth=12, max_nodes=8192)
        replay_ok, replay_detail = replay_piecewise_k3(spans, targets, *limits, result)
        return {"status": result.k3_status if replay_ok else "REPLAY_FAILED",
                "proof_replay": replay_detail, "proof_replay_pass": replay_ok,
                "nodes": result.processed_nodes, "spans": len(spans),
                "scope": "EXPLORATORY_FROZEN_MODEL_CPU_ONLY; exact serialized spline; conditional outward arithmetic"}
