#!/usr/bin/env python3
"""K4 baselines: sampled torque checks vs the rigorous continuous-time answer, per frozen output.

For each NPZ (planning-model contract + limit family given):
  truth    RIGOROUS max-utilization bracket [lb, ub] (k4_sampling): lb = rigorous point enclosure at the argmax of
           a 65,536-point uniform-s exact-model scan (+ all span knots); ub = smallest certified limit scale
           (compiled K4 producer bisection, final scale independently verified by the compiled verifier).
           K4 verdict at scale 1 (compiled producer + verifier, protocol refinement 8, escalated to 12).
  sampled  exact-model utilization at N points, N in --ns, on a uniform-s grid (TimeOpt's enforcement layout) and a
           uniform-time grid (controller/checker layout, via cuRobo's own progress table); plus native float32
           sampled maxima from k4_native_torque_samples when given (TimeOpt's own 512-node grid, S8193, S32768).
  cost     wall time of the compiled K4 producer+verifier and of each sampled check (exact model, compiled, 1 thread).
Limit families: planning = the contract's TimeOpt 0.95-scaled float32 limits; physical = effort limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NS = (8, 16, 32, 64, 128, 256, 512, 1024, 4096)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--limits", choices=("planning", "physical"), default="planning")
    ap.add_argument("--native-torque", type=Path, default=None)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--dense", type=int, default=65536)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from curobo_certify import k4_compiled as KC
    from curobo_certify import k4_sampling as KS
    from curobo_certify.k4_panda_model import PandaModel
    from curobo_certify.k4_verify import _contract_ok
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    c = json.loads(a.contract.read_text())
    if a.payload:
        from curobo_certify.k4_payload import with_payload
        c = with_payload(c, a.payload)
    if not _contract_ok(c, c["contract_sha256"]):
        raise SystemExit("contract hash mismatch")
    M = PandaModel(c)
    if a.limits == "planning":
        lo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
        hi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
    else:
        lo = np.array([float(Q(v)) for v in c["model"]["effort_lower"]])
        hi = np.array([float(Q(v)) for v in c["model"]["effort_upper"]])
    nat = {}
    if a.native_torque:
        nat = {r["npz_sha256"]: r for r in json.loads(a.native_torque.read_text())["rows"]}
    rows = []
    with tempfile.TemporaryDirectory() as d:
        b = {"producer": KC.build(Path(d) / "p.so"), "verifier": KC.build_verifier(Path(d) / "v.so")}
        P = KC.CompiledPath(M, Path(d) / "p.so")
        V = KC.CompiledVerifier(c, Path(d) / "v.so", expected_contract_sha256=c["contract_sha256"])
        CC = KC.CompiledCoefficients(M, Path(d) / "p.so")
        for f in a.npz:
            raw = f.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            with np.load(f) as z:
                g, tc = z["effective_geometry"], z["timing_controls"]
                times, prog = z["diagnostic_32768_times"], z["diagnostic_32768_progress"]
                dur = float(z["native_duration"][0])
            spans = extract_timeopt_spans(g, tc)
            T = KS.SpanTable(spans)
            # ---- K4 verdict at scale 1 (+ verifier, escalation) ----
            t0 = time.perf_counter()
            r = P.certify(g, tc, lo, hi, threads=1)
            t1 = time.perf_counter()
            ver = None
            if r["status"] == "CERTIFIED":
                ver = V.verify(g, tc, r, lo, hi, refine=8, threads=1)
            t2 = time.perf_counter()
            if ver is not None and not ver["pass"]:
                ver12 = V.verify(g, tc, r, lo, hi, refine=12, threads=a.threads)
                ver = dict(ver12, escalated_from_refine8=True)
            k4 = r["status"] if r["status"] != "CERTIFIED" else ("CERTIFIED" if ver["pass"] else "REJECTED_BY_VERIFIER")
            # ---- dense exact-model scan + rigorous bracket ----
            sd = np.unique(np.r_[KS.grid_uniform_s(a.dense), T.left, T.right])
            ud = KS.utilization(KS.sampled_torque(CC, T, sd), lo, hi)
            i, j = np.unravel_index(np.argmax(ud), ud.shape)
            lb, where = KS.rigorous_point_util(M, T, float(sd[i]), lo, hi)
            ub, ua, calls = KS.util_upper_bound(P, g, tc, lo, hi, start=max(float(ud.max()), 1e-6), threads=a.threads)
            ubv = None
            if ub is not None:
                rr = P.certify(g, tc, lo * ub, hi * ub, threads=a.threads)
                ubv = V.verify(g, tc, rr, lo * ub, hi * ub, refine=12, threads=a.threads)["pass"]
            # ---- sampled checks ----
            samp = {}
            for n in a.ns:
                for kind in ("s", "t"):
                    ss = KS.grid_uniform_s(n) if kind == "s" else KS.grid_uniform_t(n, times, prog)
                    ts = time.perf_counter()
                    u = KS.utilization(KS.sampled_torque(CC, T, ss), lo, hi)
                    samp[f"{kind}{n}"] = {"max_util": float(u.max()), "argmax_joint": int(np.argmax(u.max(0))),
                                          "seconds": time.perf_counter() - ts}
            nrow = nat.get(sha)
            row = {"path": f.stem, "npz_sha256": sha, "duration_s": dur, "spans": len(spans), "limits": a.limits,
                   "K4": k4, "k4_producer_seconds_1thread": t1 - t0, "k4_verifier_seconds_1thread": t2 - t1,
                   "k4_nodes": r["nodes"], "verifier": ver, "witness": r["witness"],
                   "dense_max_util": float(ud.max()), "dense_argmax": {"s": float(sd[i]), "joint": int(j)},
                   "truth_lb": lb, "truth_lb_where": where, "truth_ub": ub, "truth_ub_verified": ubv,
                   "truth_ub_noncertified_below": ua, "bisection_calls": calls, "sampled": samp,
                   "native_sampled": ({k: {"planning": v["max_util_planning"], "physical": v["max_util_physical"]}
                                       for k, v in nrow["grids"].items()} if nrow else None)}
            rows.append(row)
            print(json.dumps({"path": f.stem, "K4": k4, "lb": round(lb, 6), "ub": ub and round(ub, 6),
                              "s512": round(samp.get("s512", {}).get("max_util", float("nan")), 6),
                              "s64": round(samp.get("s64", {}).get("max_util", float("nan")), 6)}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "K4_BASELINES_SAMPLED_VS_RIGOROUS", "contract_sha256": c["contract_sha256"],
                                 "payload": a.payload, "limits": a.limits, "ns": a.ns, "dense": a.dense, "builds": b,
                                 "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
