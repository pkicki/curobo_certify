#!/usr/bin/env python3
"""Seeded differential/adversarial campaign: compiled K3 producer + independent
verifier versus the exact-rational Fraction reference.

Soundness relations checked on every case (any breach is a FALSE CERTIFICATE or
FALSE WITNESS and fails the campaign):
  R1 producer CERTIFIED and verifier PASS  =>  exact reference CERTIFIED
  R2 producer VIOLATED                     =>  exact residual value at the witness
                                               point is >0 (or >=0 interior for strict x)
  R3 verifier PASS                          <=  producer CERTIFIED (verifier never
                                               rejects a producer certificate here)
Also recorded: agreement/coverage rates and status confusion matrix.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import compiled_k3 as K  # noqa: E402
from curobo_certify.boundary_reference import Boundaries, derivative, evaluate  # noqa: E402
from curobo_certify.exact_k3_reference import build_residuals, certify_k3  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402

KINDS = ("random", "near_active", "broken_rest", "zero_start_speed", "zero_speed_double_root",
         "interior_zero_speed", "negative_dip", "huge", "nonfinite", "tiny_budget")


def sample_maxima(spans, n=64):
    D = len(spans[0].q)
    mv, ma, mj = np.zeros(D), np.zeros(D), np.zeros(D)
    for sp in spans:
        h = sp.right - sp.left
        u = [Fraction(i, n) for i in range(n + 1)]
        X = [[float(evaluate(derivative(sp.x, o, h) if o else sp.x, t)) for t in u] for o in range(3)]
        for j, q in enumerate(sp.q):
            Q = [[float(evaluate(derivative(q, o, h), t)) for t in u] for o in (1, 2, 3)]
            for i in range(n + 1):
                x0 = max(X[0][i], 0.0)
                mv[j] = max(mv[j], abs(Q[0][i]) * x0 ** 0.5)
                ma[j] = max(ma[j], abs(Q[1][i] * X[0][i] + 0.5 * Q[0][i] * X[1][i]))
                mj[j] = max(mj[j], x0 ** 0.5 * abs(Q[2][i] * X[0][i] + 1.5 * Q[1][i] * X[1][i] + 0.5 * Q[0][i] * X[2][i]))
    return mv, ma, mj


def make_case(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    kind = KINDS[seed % len(KINDS)] if seed % 3 else ("near_active" if seed % 2 else "random")
    D = int(rng.integers(1, 4))
    nin = int(rng.integers(1, 5))
    start, goal = rng.normal(size=D) * 2, rng.normal(size=D) * 2
    inner = [start + (goal - start) * (k + 1) / (nin + 1) + rng.normal(size=D) * rng.choice([0.01, 0.3, 1.0])
             for k in range(nin)]
    geo = np.array([start] * 3 + inner + [goal] * 3, dtype=np.float32)
    timing = (rng.uniform(0.5, 40, 32)).astype(np.float32)
    depth, nodes = 8, 4096
    if kind == "broken_rest":
        geo[2] = geo[2] + np.float32(rng.choice([1e-3, 0.5]))
    elif kind == "zero_start_speed":
        timing[0] = 0.0
    elif kind == "zero_speed_double_root":
        timing[0] = timing[1] = 0.0
    elif kind == "interior_zero_speed":
        i = int(rng.integers(4, 26))
        timing[i:i + int(rng.choice([3, 4]))] = 0.0
    elif kind == "negative_dip":
        i = int(rng.integers(3, 28))
        timing[i] = -np.float32(rng.uniform(0.1, 30))
    elif kind == "huge":
        geo = geo * np.float32(1e18)
    elif kind == "nonfinite":
        geo[int(rng.integers(0, len(geo))), 0] = rng.choice([np.nan, np.inf])
    elif kind == "tiny_budget":
        depth, nodes = int(rng.integers(0, 2)), int(rng.integers(1, 40))
    case = {"seed": seed, "kind": kind, "geometry": geo, "timing": timing, "q_start": geo[0].copy(),
            "q_goal": geo[-1].copy(), "depth": depth, "nodes": nodes}
    if kind in ("nonfinite", "huge"):
        case["limits"] = {"velocity": np.ones(D), "acceleration": np.ones(D), "jerk": np.ones(D)}
        return case
    spans = extract_timeopt_spans(geo, timing)
    mv, ma, mj = sample_maxima(spans, 32)
    f = {"near_active": lambda: rng.choice([0.995, 0.9999, 1.0001, 1.002, 1.01, 1.05]),
         }.get(kind, lambda: rng.choice([0.7, 0.98, 1.02, 1.3, 3.0]))
    case["limits"] = {k: np.maximum(m * f(), 1e-3).astype(np.float64)
                      for k, m in (("velocity", mv), ("acceleration", ma), ("jerk", mj))}
    return case


def exact_reference(case) -> str:
    try:
        spans = extract_timeopt_spans(case["geometry"], case["timing"])
    except ValueError:
        return "INVALID_INPUT"
    t = Boundaries.from_values(q_start=case["q_start"].astype(float).tolist(), q_end=case["q_goal"].astype(float).tolist(),
                               x_start=spans[0].at("x", 0)[0], x_end=spans[-1].at("x", 1)[0], strict_rest=True)
    lim = case["limits"]
    r = certify_k3(spans, t, [float(v) for v in lim["velocity"]], [float(v) for v in lim["acceleration"]],
                   [float(v) for v in lim["jerk"]], max_depth=14, max_nodes=200000)
    return r.status


def witness_ok(case, w) -> bool:
    """R2: exact residual value at the producer's witness point violates the constraint."""
    spans = extract_timeopt_spans(case["geometry"], case["timing"])
    sp = spans[w["span"]]
    lim = case["limits"]
    fam = ("positive_x", "velocity", "acceleration_plus", "acceleration_minus", "jerk")[w["family"]]
    n = 1 << w["depth"]
    left, right = Fraction(w["bits"], n), Fraction(w["bits"] + 1, n)
    u = {1: left, 2: right, 3: (left + right) / 2}[w["where"]]
    for name, joint, power in build_residuals(sp, [float(v) for v in lim["velocity"]],
                                              [float(v) for v in lim["acceleration"]], [float(v) for v in lim["jerk"]]):
        if name == fam and joint == w["joint"]:
            value = evaluate(power, u)
            if fam == "positive_x":
                interior = not ((w["span"] == 0 and u == 0) or (w["span"] == len(spans) - 1 and u == 1))
                return value >= 0 and interior
            return value > 0
    return False


_P = _V = None
_THREADS = 1


def _init(libdir, threads=1):
    global _P, _V, _THREADS
    _THREADS = threads
    _P = K.Producer(Path(libdir) / "libk3c.so")
    _V = K.Verifier(Path(libdir) / "libk3v.so")


def run_case(seed: int) -> dict:
    case = make_case(seed)
    args = (case["geometry"], case["timing"], case["q_start"], case["q_goal"], case["limits"])
    r = K.certify_and_verify(_P, _V, *args, max_depth=case["depth"], max_nodes=case["nodes"], threads=_THREADS)
    out = {"seed": seed, "kind": case["kind"], "producer": r["status"], "final": r["final"],
           "verifier": None if r["verification"] is None else r["verification"]["reason"]}
    if case["kind"] in ("nonfinite", "huge"):
        out["reference"] = "SKIPPED_NONFINITE" if case["kind"] == "nonfinite" else "SKIPPED_OVERFLOW"
    else:
        out["reference"] = exact_reference(case)
    breaches = []
    if out["final"] == "CERTIFIED" and out["reference"] not in ("CERTIFIED_K3_REFERENCE",):
        if not out["reference"].startswith("SKIPPED"):
            breaches.append("R1_false_certificate")
        else:
            breaches.append("R1_certified_nonfinite_or_overflow")
    if r["status"] == "VIOLATED" and not witness_ok(case, r["witness"]):
        breaches.append("R2_false_witness")
    if r["status"] == "CERTIFIED" and r["final"] != "CERTIFIED":
        breaches.append("R3_verifier_rejected_producer_certificate")
    if r["status"] == "VIOLATED" and out["reference"] == "CERTIFIED_K3_REFERENCE":
        breaches.append("R2_witness_contradicts_reference")
    out["breaches"] = breaches
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=10000)
    ap.add_argument("--seed0", type=int, default=20261116)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra-flags", default="")
    ap.add_argument("--threads", type=int, default=1, help="OpenMP threads per producer/verifier call")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        extra = tuple(args.extra_flags.split()) if args.extra_flags else ()
        builds = [K.build("k3_certifier.cpp", Path(d) / "libk3c.so", extra),
                  K.build("k3_verifier.cpp", Path(d) / "libk3v.so", extra)]
        t0 = time.time()
        with mp.Pool(args.workers, initializer=_init, initargs=(d, args.threads)) as pool:
            rows = pool.map(run_case, range(args.seed0, args.seed0 + args.cases), chunksize=8)
    confusion = Counter((r["producer"], r["final"], r["reference"]) for r in rows)
    breaches = [r for r in rows if r["breaches"]]
    certified_ref = sum(r["reference"] == "CERTIFIED_K3_REFERENCE" for r in rows)
    certified_both = sum(r["reference"] == "CERTIFIED_K3_REFERENCE" and r["final"] == "CERTIFIED" for r in rows)
    summary = {"scope": "SEEDED_DIFFERENTIAL_FINITE_TESTS_NOT_UNIVERSAL_PROOF", "cases": len(rows),
               "seed0": args.seed0, "builds": builds, "extra_flags": args.extra_flags,
               "elapsed_s": time.time() - t0, "threads": args.threads, "kinds": Counter(r["kind"] for r in rows),
               "breaches": len(breaches), "breach_rows": breaches[:50],
               "reference_certified": certified_ref, "compiled_also_certified": certified_both,
               "confusion": [{"producer": k[0], "final": k[1], "reference": k[2], "count": v}
                             for k, v in sorted(confusion.items(), key=lambda kv: -kv[1])]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("cases", "breaches", "reference_certified", "compiled_also_certified",
                                               "elapsed_s")}))
    for c in summary["confusion"][:25]:
        print(c)
    sys.exit(1 if breaches else 0)


if __name__ == "__main__":
    main()
