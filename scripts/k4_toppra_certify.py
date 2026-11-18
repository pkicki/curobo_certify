#!/usr/bin/env python3
"""Certify TOPP-RA parametrizations (k4_toppra_run output) in continuous time with K4 (main env).

Per (path, grid): exact spans (k4_toppra.toppra_spans), K4 verdict at TOPP-RA's own torque limits (the contract's
0.95-scaled float32 limits; Python tm1 producer + independent world-frame verifier / witness replay), rigorous
max-utilization bracket [lb, ub] (dense 65,536-point scan + rigorous point enclosure; bisection on the certified
limit scale), exact duration, and the max utilization at TOPP-RA's own gridpoints (what TOPP-RA enforced).
Velocity/acceleration are reported as exact-model dense maxima (diagnostic).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toppra", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--dense", type=int, default=65536)
    ap.add_argument("--max-depth", type=int, default=14)
    ap.add_argument("--max-nodes", type=int, default=200000)
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import tempfile
    from curobo_certify import k4_compiled as KC
    from curobo_certify import k4_sampling as KS
    from curobo_certify.k4_panda_model import PandaModel
    from curobo_certify.k4_reuse import GeometryCache
    from curobo_certify.k4_toppra import toppra_duration, toppra_spans
    from curobo_certify.k4_verify import _contract_ok, verify_k4, verify_k4_witness
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    c = json.loads(a.contract.read_text())
    if a.payload:
        from curobo_certify.k4_payload import with_payload
        c = with_payload(c, a.payload)
    if not _contract_ok(c, c["contract_sha256"]):
        raise SystemExit("contract hash mismatch")
    tp = json.loads(a.toppra.read_text())
    if tp["contract_sha256"] != c["contract_sha256"]:
        raise SystemExit("TOPP-RA run used a different contract")
    M = PandaModel(c)
    lo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
    hi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
    tmp = tempfile.TemporaryDirectory()
    KC.build(Path(tmp.name) / "p.so")
    CC = KC.CompiledCoefficients(M, Path(tmp.name) / "p.so")
    rows = []
    todo = [r for i, r in enumerate(tp["rows"]) if i % a.shard[1] == a.shard[0]]
    for rec in todo:
        out = {k: rec.get(k) for k in ("path", "grid", "ok", "seconds", "native_duration_s", "return_code", "error")}
        if not rec.get("ok"):
            rows.append(out | {"K4": "TOPPRA_FAILED"})
            continue
        with np.load(a.npz_dir / f"{rec['path']}.npz") as z:
            base = extract_timeopt_spans(z["effective_geometry"], z["timing_controls"])
        if tp.get("trim"):
            from curobo_certify.k4_toppra import trimmed
            base, _ = trimmed(base)
        grid = [Q(float.fromhex(h)) for h in rec["gridpoints_hex"]]
        sd = [Q(float.fromhex(h)) for h in rec["sd_hex"]]
        grid[0], grid[-1] = Q(0), Q(1)
        spans = toppra_spans(base, grid, [v * v for v in sd])
        dur = toppra_duration(grid, sd)
        T = KS.SpanTable(spans)
        t0 = time.perf_counter()
        C = GeometryCache(M, spans, "tm1")
        r = C.certify(spans, lo, hi, max_depth=a.max_depth, max_nodes=a.max_nodes)
        t1 = time.perf_counter()
        if r.status == "CERTIFIED":
            v = verify_k4(spans, c, r, lo, hi, expect_physical=False, refine=8, expected_contract_sha256=c["contract_sha256"])
            if not v["pass"]:
                v = dict(verify_k4(spans, c, r, lo, hi, expect_physical=False, refine=12, expected_contract_sha256=c["contract_sha256"]), escalated=True)
        elif r.status == "VIOLATED_TORQUE":
            v = verify_k4_witness(spans, c, r, lo, hi, expected_contract_sha256=c["contract_sha256"])
        else:
            v = {"pass": None}
        k4 = r.status if r.status not in ("CERTIFIED", "VIOLATED_TORQUE") or v["pass"] else "REJECTED_BY_VERIFIER"
        # dense scan + rigorous lower bound
        sdense = np.unique(np.r_[KS.grid_uniform_s(a.dense), T.left, T.right])
        ud = KS.utilization(KS.sampled_torque(CC, T, sdense), lo, hi)
        i = int(np.argmax(ud.max(1)))
        lb, where = KS.rigorous_point_util(M, T, float(sdense[i]), lo, hi)
        # what TOPP-RA enforced: utilization at its own gridpoints (both one-sided values via the two adjacent spans)
        gp = np.array([float(x) for x in grid])
        ug_left = KS.utilization(KS.sampled_torque(CC, T, gp), lo, hi).max()
        # rigorous upper bound by bisection on the certified scale (geometry prepared once)
        def cert(f):
            return C.certify(spans, lo * f, hi * f).status == "CERTIFIED"
        lo_f, hi_f, calls = lb * 0.999, max(lb, 1e-6), 0
        while not cert(hi_f) and calls < 30:
            lo_f, hi_f = hi_f, hi_f * 1.01 + 1e-4
            calls += 1
        while hi_f - lo_f > 1e-6 and calls < 60:
            m = 0.5 * (lo_f + hi_f)
            calls += 1
            if cert(m):
                hi_f = m
            else:
                lo_f = m
        q, q1, q2, x, x1 = T.evaluate(sdense)
        vel = np.abs(q1 * np.sqrt(np.maximum(x, 0))[:, None]).max(0)
        acc = np.abs(q2 * x[:, None] + 0.5 * q1 * x1[:, None]).max(0)
        out.update({"K4": k4, "producer": r.status, "verifier_pass": v.get("pass"), "nodes": r.nodes, "depth": r.max_depth,
                    "witness": r.witness, "certify_seconds": t1 - t0, "n_spans": len(spans), "n_grid": len(grid),
                    "duration_s": float(dur) if dur is not None else None,
                    "truth_lb": lb, "truth_lb_where": where, "truth_ub": hi_f if cert(hi_f) else None,
                    "dense_max_util": float(ud.max()), "gridpoint_max_util": float(ug_left),
                    "dense_max_abs_velocity": vel.tolist(), "dense_max_abs_acceleration": acc.tolist()})
        rows.append(out)
        print(json.dumps({k: out[k] for k in ("path", "grid", "K4", "n_grid", "duration_s", "truth_lb", "gridpoint_max_util")}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "TOPPRA_BASELINE_K4", "contract_sha256": c["contract_sha256"], "payload": a.payload,
                                 "trim": tp.get("trim"), "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
