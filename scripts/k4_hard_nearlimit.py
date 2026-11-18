#!/usr/bin/env python3
"""K4 HARD CASES (1): near-limit decision ladder - where do sampled checks stop deciding correctly?

From each base output (fixed geometry), uniformly time-scaled variants are built: the 32 float32 timing controls are
multiplied by lambda^2 (x -> ~lambda^2 x, time -> time/lambda), rounded to float32; the resulting bytes ARE the case
(spans re-extracted exactly from them). lambda is chosen so that the exact-model peak planning-limit utilization is
1 + eps for eps in the ladder (both signs). Per case:
  truth     RIGOROUS bracket [lb, ub]: lb = rigorous point enclosure at a locally refined argmax; ub = smallest
            certified limit scale (compiled K4 bisection, tolerance 1e-11, deep budgets), final scale verified by the
            compiled verifier at refinement 12.
  K4        compiled producer + verifier at scale 1 with the protocol budget (depth 12 / 65,536 nodes) AND a deep
            budget (depth 24 / 4,000,000 nodes): CERTIFIED / VIOLATED_TORQUE / UNKNOWN (+ seconds).
  sampled   exact-model float64 checks, N in {512, 4096, 32768}, uniform-time (cuRobo progress table, times/lambda)
            and uniform-s. (Native float32 checks on the same bytes: k4_hard_native.py, GPU.)
A, B, G depend on geometry only, so they are computed once per base path on all sample/dense points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LADDER = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 3e-7, -3e-7, -1e-6, -1e-5, -1e-4, -1e-3, -1e-2)
NS = (512, 4096, 32768)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--eps", type=float, action="append", default=None, help="repeat --eps=VALUE (negative allowed)")
    ap.add_argument("--dense", type=int, default=131072)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from curobo_certify import k4_compiled as KC
    from curobo_certify import k4_sampling as KS
    from curobo_certify.k4_interval import IA
    from curobo_certify.k4_panda_model import PandaModel
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    c = json.loads(a.contract.read_text())
    if a.payload:
        from curobo_certify.k4_payload import with_payload
        c = with_payload(c, a.payload)
    M = PandaModel(c)
    lo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
    hi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
    tmp = tempfile.TemporaryDirectory()
    build = {"producer": KC.build(Path(tmp.name) / "p.so"), "verifier": KC.build_verifier(Path(tmp.name) / "v.so")}
    P = KC.CompiledPath(M, Path(tmp.name) / "p.so")
    V = KC.CompiledVerifier(c, Path(tmp.name) / "v.so", expected_contract_sha256=c["contract_sha256"])
    CC = KC.CompiledCoefficients(M, Path(tmp.name) / "p.so")

    def abg(T, s):
        q, q1, q2, _, _ = T.evaluate(s)
        box = lambda x: [IA(x[:, j].copy()) for j in range(7)]  # noqa: E731
        A, B, G = CC.coefficients(box(q), box(q1), box(q2))
        mid = lambda L: np.stack([0.5 * (v.lo + v.hi) for v in L], 1)  # noqa: E731
        return mid(A), mid(B), mid(G)

    def util(ABG, x, x1):
        A, B, G = ABG
        tau = A * x[:, None] + B * x1[:, None] + G
        return np.maximum(tau / hi, tau / lo).max(1)

    rows = []
    for f in a.npz:
        with np.load(f) as z:
            g, tc0 = z["effective_geometry"], z["timing_controls"].astype(np.float32)
            times, prog = z["diagnostic_32768_times"].astype(float), z["diagnostic_32768_progress"].astype(float)
        T0 = KS.SpanTable(extract_timeopt_spans(g, tc0))
        sd = np.unique(np.r_[np.linspace(0, 1, a.dense), T0.left, T0.right])
        pts = {"dense": sd}
        for n in NS:
            pts[f"s{n}"] = np.linspace(0, 1, n)
            pts[f"t{n}"] = np.interp(np.linspace(0, times[-1], n), times, prog)   # same s points for every lambda
        t0 = time.perf_counter()
        cache = {k: abg(T0, v) for k, v in pts.items()}
        prep = time.perf_counter() - t0
        # base peak decomposition: U(l2) = max_s max_j (l2 * P + G)/lim  with P = A x + B x' of the base timing
        _, _, _, x0, x10 = T0.evaluate(sd)
        A, B, G = cache["dense"]
        Pd = A * x0[:, None] + B * x10[:, None]

        def U_of(l2):
            tau = l2 * Pd + G
            return float(np.maximum(tau / hi, tau / lo).max())
        for eps in (a.eps or LADDER):
            target = 1 + eps
            lo_l, hi_l = 1e-4, 1.0
            while U_of(hi_l) < target:
                hi_l *= 2
                if hi_l > 1e6:
                    break
            for _ in range(200):
                m = 0.5 * (lo_l + hi_l)
                if U_of(m) < target:
                    lo_l = m
                else:
                    hi_l = m
            l2 = 0.5 * (lo_l + hi_l)
            tc = (tc0 * np.float32(l2)).astype(np.float32)
            spans = extract_timeopt_spans(g, tc)
            T = KS.SpanTable(spans)
            # exact sampled checks on the actual bytes
            samp = {}
            for k, sp in pts.items():
                _, _, _, x, x1 = T.evaluate(sp)
                u = util(cache[k], x, x1)
                samp[k] = float(u.max())
                if k == "dense":
                    iarg = int(np.argmax(u))
            # local refinement of the argmax (float64), then a rigorous point enclosure -> lb
            s_star, width = sd[iarg], 2.0 / a.dense
            for _ in range(4):
                loc = np.clip(np.linspace(s_star - width, s_star + width, 2049), 0, 1)
                _, _, _, x, x1 = T.evaluate(loc)
                u = util(abg(T, loc), x, x1)
                s_star, width = loc[int(np.argmax(u))], width / 512
            lb, where = KS.rigorous_point_util(M, T, float(s_star), lo, hi)
            # K4 at scale 1: protocol and deep budgets
            k4 = {}
            for name, (dep, nod) in (("protocol", (12, 65536)), ("deep", (24, 4_000_000))):
                ts = time.perf_counter()
                r = P.certify(g, tc, lo, hi, max_depth=dep, max_nodes=nod, threads=a.threads)
                st = r["status"]
                if st == "CERTIFIED":
                    v = V.verify(g, tc, r, lo, hi, refine=8, threads=a.threads)
                    if not v["pass"]:
                        v = V.verify(g, tc, r, lo, hi, refine=12, threads=a.threads)
                    st = "CERTIFIED" if v["pass"] else "REJECTED_BY_VERIFIER"
                k4[name] = {"status": st, "nodes": r["nodes"], "depth": r["max_depth"], "seconds": time.perf_counter() - ts,
                            "witness": r["witness"]}
            # rigorous upper bound (deep budget bisection on the limit scale)
            def cert(fac):
                return P.certify(g, tc, lo * fac, hi * fac, max_depth=24, max_nodes=4_000_000, threads=a.threads)["status"] == "CERTIFIED"
            fa, fb, calls = lb * (1 - 1e-6), lb * (1 + 1e-6), 0
            while not cert(fb) and calls < 20:
                fa, fb = fb, fb * (1 + 1e-5); calls += 1
            while fb - fa > 1e-11 and calls < 80:
                mm = 0.5 * (fa + fb); calls += 1
                if cert(mm):
                    fb = mm
                else:
                    fa = mm
            rr = P.certify(g, tc, lo * fb, hi * fb, max_depth=24, max_nodes=4_000_000, threads=a.threads)
            ubv = rr["status"] == "CERTIFIED" and V.verify(g, tc, rr, lo * fb, hi * fb, refine=12, threads=a.threads)["pass"]
            truth = "VIOLATED" if lb > 1 else ("CERTIFIED" if (ubv and fb <= 1) else "UNDECIDED")
            row = {"base": f.stem, "eps_target": eps, "lambda2": l2, "time_scale": float(1 / np.sqrt(l2)),
                   "timing_controls_hex": [float(v).hex() for v in tc],
                   "tc_sha256": hashlib.sha256(tc.tobytes()).hexdigest(),
                   "truth": truth, "lb": lb, "ub": fb if ubv else None, "lb_where": where, "dense_U": samp["dense"],
                   "sampled": {k: v for k, v in samp.items() if k != "dense"}, "K4": k4, "prep_seconds": prep}
            rows.append(row)
            print(json.dumps({"base": f.stem[-18:], "eps": eps, "truth": truth, "lb-1": lb - 1, "ub-1": (fb - 1) if ubv else None,
                              "K4": k4["protocol"]["status"], "K4deep": k4["deep"]["status"],
                              "t32768-1": samp["t32768"] - 1, "s32768-1": samp["s32768"] - 1}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "K4_HARD_NEARLIMIT_LADDER", "contract_sha256": c["contract_sha256"],
                                 "payload": a.payload, "limits": "planning (0.95-scaled float32)", "builds": build,
                                 "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
