#!/usr/bin/env python3
"""K4 P3 comparison: direct interval RNEA vs cached geometry boxes vs certified Taylor models (tm1),
adaptive vs uniform subdivision, depth/node budgets, over real geometries x timing variants.

Timing variants on the SAME geometry: x_k(s) = k^2 x(s) (uniform time scaling by 1/k), k in --speeds.
For every (path, variant) and method: status, nodes, RNEA boxes, preparation (geometry) seconds, per-
retiming seconds, certified worst utilisation bound, and dense rigorous point maximum (conservatism =
bound - point maximum; the point maximum is a sampled diagnostic, the bound is rigorous). Python CPU
reference timings (1 thread); not production latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402
from curobo_certify.k4_path import _point_boxes, certify_path, torque_enclosure  # noqa: E402
from curobo_certify.k4_reference import dpoly  # noqa: E402
from curobo_certify.k4_reuse import GeometryCache  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


def dense_max_util(M, spans, lo, hi, per_span=24):
    pts, rl = [], []
    for sp in spans:
        h = sp.right - sp.left
        raw = ([tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q]
               + [tuple(sp.x), dpoly(sp.x, 1, h)])
        for t in range(per_span):
            pts.append(Q(t, per_span - 1)); rl.append(raw)
    pq, pqp, pqpp, px, pxp, _ = _point_boxes(rl, pts, 7)
    tau = torque_enclosure(M, pq, pqp, pqpp, px, pxp)
    T = np.stack([0.5 * (t.lo + t.hi) for t in tau], 1)
    return np.maximum(T / hi, T / lo).max(0)


def uniform(M, spans, lo, hi, depth):
    """Uniform depth-d subdivision, no adaptivity (all 2^d leaves per span)."""
    from curobo_certify.k4_path import _boxes, _raw_polys
    from curobo_certify.k4_reference import bernstein, split
    t0 = time.perf_counter()
    leaves = []
    for si, sp in enumerate(spans):
        cur = [tuple(bernstein(p) for p in _raw_polys(sp, 7))]
        for _ in range(depth):
            cur = [tuple(split(p)[s] for p in b) for b in cur for s in (0, 1)]
        leaves += cur
    q, qp, qpp, x, xp = _boxes(leaves, 7)
    tau = torque_enclosure(M, q, qp, qpp, x, xp)
    tlo = np.stack([t.lo for t in tau], 1); thi = np.stack([t.hi for t in tau], 1)
    ok = bool(((tlo >= lo).all() and (thi <= hi).all() and (x.lo >= 0).all()))
    util = np.maximum(np.where(thi > 0, thi / hi, 0), np.where(tlo < 0, tlo / lo, 0)).max(0)
    return ("CERTIFIED" if ok else "UNKNOWN_UNIFORM"), len(leaves), float(util.max()), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--contract", type=Path, default=ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
    ap.add_argument("--speeds", nargs="+", type=float, default=[1.0, 1.1, 1.2, 1.3, 1.5])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = load_contract(a.contract)
    M = PandaModel(c)
    lo = np.array([float(Q(v)) for v in c["model"]["effort_lower"]])
    hi = np.array([float(Q(v)) for v in c["model"]["effort_upper"]])
    rows = []
    for f in a.npz:
        z = np.load(f)
        g, tc = z["effective_geometry"], z["timing_controls"]
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        base = extract_timeopt_spans(g, tc)
        caches = {}
        for mode in ("box", "tm1"):
            t0 = time.perf_counter()
            caches[mode] = GeometryCache(M, base, mode)
            caches[mode].ensure([(si, "") for si in range(len(base))])
            caches[mode].init_seconds = time.perf_counter() - t0
        for k in a.speeds:
            sp = base if k == 1.0 else extract_timeopt_spans(g, (tc * np.float32(k * k)).astype(np.float32))
            pm = dense_max_util(M, sp, lo, hi)
            row = {"path": f.stem, "npz_sha256": sha, "speed": k, "dense_point_max_util": float(pm.max()),
                   "dense_point_max_util_per_joint": pm.tolist(), "methods": {}}
            r = certify_path(sp, M, lo, hi)
            row["methods"]["direct_adaptive"] = {"status": r.status, "nodes": r.nodes, "depth": r.max_depth,
                                                 "bound": max(r.worst_utilization or [0]), "seconds": r.seconds,
                                                 "witness": r.witness}
            for mode, C in caches.items():
                p0, b0 = C.prep_seconds, C.rnea_boxes
                t0 = time.perf_counter()
                r = C.certify(sp, lo, hi)
                row["methods"][f"cache_{mode}"] = {"status": r.status, "nodes": r.nodes, "depth": r.max_depth,
                                                   "bound": max(r.worst_utilization or [0]),
                                                   "seconds": time.perf_counter() - t0,
                                                   "prep_seconds_in_call": C.prep_seconds - p0,
                                                   "new_rnea_boxes": C.rnea_boxes - b0}
            for d in (0, 1, 2, 4):
                st, n, b, s = uniform(M, sp, lo, hi, d)
                row["methods"][f"uniform_d{d}"] = {"status": st, "nodes": n, "bound": b, "seconds": s}
            if k == a.speeds[0]:
                for d in (0, 1, 2, 4, 8):
                    r = certify_path(sp, M, lo, hi, max_depth=d)
                    row["methods"][f"adaptive_depth{d}"] = {"status": r.status, "nodes": r.nodes,
                                                            "bound": max(r.worst_utilization or [0]), "seconds": r.seconds}
                for nn in (2048, 8192, 32768):
                    r = certify_path(sp, M, lo, hi, max_nodes=nn)
                    row["methods"][f"adaptive_nodes{nn}"] = {"status": r.status, "nodes": r.nodes, "seconds": r.seconds}
            rows.append(row)
            print(json.dumps({"path": f.stem, "k": k, "pm": round(row["dense_point_max_util"], 4),
                              **{m: (v["status"], round(v.get("bound", 0), 4)) for m, v in row["methods"].items()
                                 if m in ("direct_adaptive", "cache_box", "cache_tm1", "uniform_d0")}}), flush=True)
        for mode, C in caches.items():
            rows[-1].setdefault("geometry_prep", {})[mode] = {"init_seconds": C.init_seconds,
                                                              "total_prep_seconds": C.prep_seconds,
                                                              "rnea_boxes": C.rnea_boxes}
    out = {"scope": "K4_P3_EXPLORATORY_COMPARISON_PYTHON_CPU", "contract_sha256": c["contract_sha256"],
           "speeds": a.speeds, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
