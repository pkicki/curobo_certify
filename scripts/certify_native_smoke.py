#!/usr/bin/env python3
"""Certify the frozen synthetic native smoke polynomial model (not a cohort)."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curobo_certify.boundary_reference import Boundaries  # noqa: E402
from curobo_certify.piecewise_cpu import (  # noqa: E402
    certify_piecewise_k3,
    exact_oracle_status,
    replay_piecewise_k3,
)
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


def main() -> int:
    source_json = ROOT / "runs/native_adapter_parity.json"
    source_npz = ROOT / "runs/native_adapter_parity_arrays.npz"
    meta = json.loads(source_json.read_text())
    raw_arrays = source_npz.read_bytes()
    arrays_sha = hashlib.sha256(raw_arrays).hexdigest()
    if meta.get("status") != "PASS" or meta.get("frozen_arrays", {}).get("sha256") != arrays_sha:
        raise SystemExit("native smoke parity/hash gate did not pass")
    if meta.get("evidence_kind") != "synthetic_native_adapter_parity_smoke":
        raise SystemExit("source artifact is not explicitly synthetic smoke data")

    with np.load(source_npz, allow_pickle=False) as arrays:
        spans = extract_timeopt_spans(arrays["effective_geometry"][0].copy(),
                                      arrays["timing_controls"][0].copy())
        start = Fraction.from_float(float(arrays["requested_start"][0, 0]))
        end = Fraction.from_float(float(arrays["requested_goal"][0, 0]))
        limits = arrays["physical_limits"].astype(np.float64)
        planning = arrays["planning_scaled_limits"].astype(np.float64)
    targets = Boundaries.from_values(
        q_start=[start], q_end=[end], x_start=spans[0].at("x", 0)[0],
        x_end=spans[-1].at("x", 1)[0],
    )
    values = (limits[0].tolist(), limits[1].tolist(), limits[2].tolist())
    result = certify_piecewise_k3(spans, targets, *values, max_depth=12, max_nodes=8192)
    replay_ok, replay_detail = replay_piecewise_k3(spans, targets, *values, result)
    oracle_status = exact_oracle_status(spans, targets, *values, max_depth=12, max_nodes=8192)
    output = {
        "evidence_kind": "synthetic_native_smoke_polynomial_certificate",
        "campaign_data": False,
        "source_id": meta["source_id"],
        "scene_id": meta["scene_id"],
        "upstream_sha": meta["upstream_sha"],
        "source_json_sha256": hashlib.sha256(source_json.read_bytes()).hexdigest(),
        "source_arrays_sha256": arrays_sha,
        "model_urdf_sha256": meta["model"]["urdf_sha256"],
        "native_parity_status": meta["status"],
        "span_count": len(spans),
        "physical_limits": {"velocity": values[0], "acceleration": values[1], "jerk": values[2]},
        "planning_scaled_limits": {
            "velocity": planning[0].tolist(), "acceleration": planning[1].tolist(), "jerk": planning[2].tolist()
        },
        "k3_status": result.k3_status,
        "full_contract_status": result.full_contract_status,
        "boundary_status": result.boundary_status,
        "exact_fraction_oracle_status": oracle_status,
        "outward_proof_replay_pass": replay_ok,
        "outward_proof_replay_detail": replay_detail,
        "processed_nodes": result.processed_nodes,
        "proof_scope": "exact extracted float32 spline polynomial model; outward CPU K3 under documented IEEE assumptions",
        "unsupported_families": ["torque", "torque_rate", "collision", "full robot dynamics"],
        "reason": result.reason,
    }
    out = ROOT / "runs/native_smoke_k3_certificate.json"
    out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if (result.k3_status == "CERTIFIED_K3" and replay_ok
                  and oracle_status == "CERTIFIED_K3_REFERENCE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
