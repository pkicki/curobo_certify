#!/usr/bin/env python3
"""Deterministic synthetic K4 correctness pilot; no Panda or native torque claims.

Example:
 PYTHONPATH=src python scripts/run_k4_reference_pilot.py --out /tmp/k4_synthetic.json
All timings are PYTHON EXACT-RATIONAL REFERENCE costs, never production latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curobo_certify.k4_reference import (Pendulum, PlanarTwoLink, PathSpan,
                                        certify_torque, replay_torque_leaves)


def examples():
    m = {"inertia": "1", "mass": "1", "com": "1", "gravity": "1"}
    two = {"m1": "1", "m2": "1", "l1": "1", "c1": "1/2", "c2": "1/2",
           "J1": "0", "J2": "0", "gravity": "0"}
    p = [{"left": "0", "right": "1", "q": [["0", "1"]], "x": ["1"]}]
    p2 = [{"left": "0", "right": "1", "q": [["0", "0", "1/2"], ["0"]], "x": ["1"]}]
    return [
        {"id": "pendulum_sin_feasible", "model": "pendulum", "parameters": m,
         "spans": p, "torque_limits": [["-9/10", "9/10"]]},
        {"id": "pendulum_sin_violated", "model": "pendulum", "parameters": m,
         "spans": p, "torque_limits": [["-1/2", "1/2"]]},
        {"id": "two_link_accel_feasible", "model": "two_link", "parameters": two,
         "spans": p2, "torque_limits": [["-3", "3"], ["-1", "1"]]},
        {"id": "two_link_accel_violated", "model": "two_link", "parameters": two,
         "spans": p2, "torque_limits": [["-2", "2"], ["-1", "1"]]},
        {"id": "two_link_slow_timing", "model": "two_link", "parameters": two,
         "spans": [{**p2[0], "x": ["1/2"]}],
         "torque_limits": [["-2", "2"], ["-1", "1"]]},
    ]


def _load_case(case):
    if case.get("model") == "pendulum":
        model = Pendulum(**case["parameters"])
    elif case.get("model") == "two_link":
        model = PlanarTwoLink(**case["parameters"])
    else:
        raise ValueError("only analytic pendulum/two_link models are supported")
    spans = [PathSpan.from_values(s["left"], s["right"], s["q"], s["x"])
             for s in case["spans"]]
    return model, spans, case["torque_limits"]


def run(cases, depths, warmups, repeats):
    rows = []
    for c in cases:
        model, spans, limits = _load_case(c)
        for depth in depths:
            def execute():
                result = certify_torque(spans, model, limits, max_depth=depth)
                replay = replay_torque_leaves(spans, model, limits, result) if result.status == "CERTIFIED" else None
                if result.status == "CERTIFIED" and not replay:
                    raise RuntimeError("REPLAY_REJECTED certificate")
                return result, replay
            for _ in range(warmups):
                execute()
            costs = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                result, replay = execute()
                costs.append((time.perf_counter_ns()-start)/1e6)
            row = {"id": c["id"], "depth": depth, "status": result.status,
                   "scope": result.scope, "nodes": result.nodes, "leaves": len(result.leaves),
                   "replay_pass": replay, "witness": result.to_json()["witness"],
                   "cost_ms": {"median": statistics.median(costs), "max": max(costs),
                               "observations": costs}}
            rows.append(row)
            print(f'{row["id"]}: depth={depth} {row["status"]}, leaves={row["leaves"]}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional JSON list of synthetic analytical fixtures")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[0, 2, 4, 8, 12])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    a = parser.parse_args()
    if a.out.exists() or a.warmups < 0 or a.repeats < 1 or any(d < 0 for d in a.depths):
        raise SystemExit("invalid budgets or output exists; never overwrite results")
    if a.input:
        raw = a.input.read_bytes()
        cases = json.loads(raw)
    else:
        cases = examples()
        raw = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    if not isinstance(cases, list) or not cases:
        raise SystemExit("expected nonempty JSON array")
    rows = run(cases, a.depths, a.warmups, a.repeats)
    payload = {"scope": "EXPLORATORY_SYNTHETIC_MODEL_TORQUE_REFERENCE_ONLY",
               "not_claimed": ["Panda", "URDF/RNEA", "native_FP32_torque", "online_latency",
                               "full_K3+K4", "torque_rate", "motor_torque"],
               "fixtures_sha256": hashlib.sha256(raw).hexdigest(),
               "reference_sha256": hashlib.sha256((Path(__file__).resolve().parents[1]/"src/curobo_certify/k4_reference.py").read_bytes()).hexdigest(),
               "warmups": a.warmups, "repeats": a.repeats, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
