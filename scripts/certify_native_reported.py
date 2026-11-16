#!/usr/bin/env python3
"""Native-reported K3 certificates for frozen native TimeOpt outputs.

For each frozen NPZ: compute the rigorous per-common-span native evaluation
error bounds (``curobo_certify.native_error_bound``) and certify the exact model
against original physical limits reduced by those bounds, with exact boundary
and knot checks and exact Bernstein subdivision.

Claim when CERTIFIED (scope NATIVE_REPORTED_K3_CONDITIONAL_ON_DOCUMENTED_FP32_SEMANTICS):
for every float32 progress s in [0, 1+4u], hence for every finite requested
time, the pinned native sample kernel reports velocity/acceleration/jerk within
the original physical limits, conditional on the ArithmeticModel assumptions.
Not claimed: time consistency of s(t), torque, collision, controller tracking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import native_error_bound as NEB  # noqa: E402
from curobo_certify.boundary_reference import Boundaries  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402

TABLE = ROOT / "fixtures/timeopt_timing_polynomial_c301cb.npy"


def certify_npz(path: Path, model: NEB.ArithmeticModel, pieces: int, depth: int, nodes: int) -> dict:
    raw = path.read_bytes()
    t0 = time.perf_counter()
    with np.load(path, allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    eff, tc = arrays["effective_geometry"], arrays["timing_controls"]
    before = hashlib.sha256(eff.tobytes() + tc.tobytes()).hexdigest()
    spans = extract_timeopt_spans(eff, tc)
    limits = {name: np.minimum(np.abs(arrays[key][0]), np.abs(arrays[key][1])).astype(float).tolist()
              for name, key in (("velocity", "velocity_limits"), ("acceleration", "acceleration_limits"),
                                ("jerk", "jerk_limits"))}
    targets = Boundaries.from_values(q_start=arrays["requested_start"][0].astype(float).tolist(),
                                     q_end=arrays["requested_goal"][0].astype(float).tolist(),
                                     x_start=spans[0].at("x", 0)[0], x_end=spans[-1].at("x", 1)[0],
                                     strict_rest=True)
    table = np.load(TABLE)
    bounds = NEB.common_span_bounds(model, spans, eff.tolist(), tc.tolist(), table, pieces=pieces)
    t1 = time.perf_counter()
    status, detail, nproc, leaves = NEB.certify_native_reported(spans, targets, limits, bounds,
                                                                max_depth=depth, max_nodes=nodes)
    t2 = time.perf_counter()
    after = hashlib.sha256(eff.tobytes() + tc.tobytes()).hexdigest()
    rel = {fam: max(getattr(b, fam) / limits[fam][b.joint] for row in bounds for b in row)
           for fam in ("velocity", "acceleration", "jerk")}
    worst = {fam: max(((getattr(b, fam) / limits[fam][b.joint], b.span_index, b.joint) for row in bounds for b in row))
             for fam in ("velocity", "acceleration", "jerk")}
    return {
        "path": path.stem, "source_npz_sha256": hashlib.sha256(raw).hexdigest(),
        "inputs_unchanged": before == after, "native_duration_s": float(arrays["native_duration"][0]),
        "common_spans": len(spans), "joints": len(spans[0].q),
        "status": status, "detail": detail, "nodes": nproc, "leaves": len(leaves),
        "max_relative_bound": rel,
        "worst_relative_bound_location": {k: {"ratio": v[0], "span": v[1], "joint": v[2]} for k, v in worst.items()},
        "per_span_joint_bounds": [[[b.velocity, b.acceleration, b.jerk] for b in row] for row in bounds],
        "timing_ms": {"bounds": 1e3 * (t1 - t0), "tightened_exact_certificate": 1e3 * (t2 - t1)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--pieces", type=int, default=16)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--nodes", type=int, default=65536)
    ap.add_argument("--div-ulps", type=float, default=2.0)
    ap.add_argument("--sqrt-ulps", type=float, default=3.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    model = NEB.ArithmeticModel(div_ulps=args.div_ulps, sqrt_ulps=args.sqrt_ulps)
    rows = []
    for p in args.npz:
        r = certify_npz(p, model, args.pieces, args.depth, args.nodes)
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("path", "status", "max_relative_bound", "nodes", "timing_ms")}))
    out = {"scope": NEB.NATIVE_SCOPE, "arithmetic_model": model.describe(),
           "pieces_per_span": args.pieces, "depth": args.depth, "nodes": args.nodes,
           "timing_table_sha256": hashlib.sha256(np.load(TABLE).tobytes()).hexdigest(),
           "bound_module_sha256": hashlib.sha256((ROOT / "src/curobo_certify/native_error_bound.py").read_bytes()).hexdigest(),
           "not_claimed": ["time consistency of native s(t) inversion", "torque/torque-rate", "collision",
                           "controller tracking", "outputs of failed or stationary native requests"],
           "rows": rows,
           "certified": sum(r["status"] == "CERTIFIED" for r in rows), "denominator": len(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"NATIVE_REPORTED certified {out['certified']}/{out['denominator']} -> {args.out}")


if __name__ == "__main__":
    main()
