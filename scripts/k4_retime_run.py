#!/usr/bin/env python3
"""K4 P6: certification-aware retiming experiment on frozen native outputs (fixed geometry).

Per output: build tm1 geometry cache and sufficient linear velocity/acceleration/torque constraints on
the timing controls (k4_retime), minimise the convex duration objective, round to float32 controls and
accept ONLY a candidate that passes the full MODEL_K3 (frozen compiled producer + verifier, physical
limits, incl. jerk/positivity/boundary) and MODEL_K4 (tm1 + independent world-frame verifier) proofs;
otherwise bisect towards the native controls (lam in [0, 1], 10 steps). Durations: rigorous time-law
bracket (time_law_grid, 2^16 cells) for native and accepted controls. Reports the gain vs native 0.95
(the paired native 0.99 comparison is computed in k4_analyze_population.py). With --limit-scale 0.95
the retimed timing obeys the same 0.95 planning margin as native TimeOpt (fair comparison; review N-P6-1).
Exploratory; Python CPU.
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
sys.path.insert(0, str(ROOT / "scripts"))
from curobo_certify import compiled_k3 as K  # noqa: E402
from curobo_certify import time_law_grid as TG  # noqa: E402
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402
from curobo_certify.k4_retime import sampled_constraints, slp_optimize  # noqa: E402
from curobo_certify.k4_reuse import GeometryCache  # noqa: E402
from curobo_certify.k4_verify import verify_k4  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402
from bench_certification_methods import limits_of  # noqa: E402


def full_certify(P, V, M, c, arr, geo, tc, lo, hi, cache, klim, physical=True):
    args = (geo, tc, arr["requested_start"][0], arr["requested_goal"][0], klim)
    r3 = P.certify(*args, max_depth=12, max_nodes=8192)
    k3 = r3["status"]
    if k3 == "CERTIFIED":
        k3 = "CERTIFIED" if V.verify(*args, r3["leaves"], K.input_digest(*args, None))["pass"] else "REJECTED_BY_VERIFIER"
    spans = extract_timeopt_spans(geo, tc)
    r4 = cache.certify(spans, lo, hi)
    k4 = r4.status
    if k4 == "CERTIFIED" and not verify_k4(spans, c, r4, lo, hi, refine=8, expect_physical=physical)["pass"]:
        k4 = "REJECTED_BY_VERIFIER"
    return k3, k4


def duration(tc):
    g = TG.build_grid(np.asarray(tc, dtype=np.float32), 16)
    return list(g.duration) if g.status == "BOUNDED_EXACT_MODEL" else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--contract", type=Path, default=ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
    ap.add_argument("--limit-scale", choices=("1.00", "0.95"), default="1.00",
                    help="1.00: physical limits; 0.95: the SAME planning margin native TimeOpt uses (K3 v/a/j and torque "
                         "scaled like cuRobo, float32) - gains then measure timing quality, not consumed margin")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = load_contract(a.contract)
    M = PandaModel(c)
    physical = a.limit_scale == "1.00"
    if physical:
        lo = np.array([float(Q(v)) for v in c["model"]["effort_lower"]])
        hi = np.array([float(Q(v)) for v in c["model"]["effort_upper"]])
    else:
        lo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
        hi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
    rows = []
    with tempfile.TemporaryDirectory() as d:
        K.build("k3_certifier.cpp", Path(d) / "c.so"); K.build("k3_verifier.cpp", Path(d) / "v.so")
        P, V = K.Producer(Path(d) / "c.so"), K.Verifier(Path(d) / "v.so")
        for f in a.npz:
            t0 = time.perf_counter()
            with np.load(f) as z:
                arr = {k: z[k].copy() for k in z.files}
            geo, tc0 = arr["effective_geometry"], arr["timing_controls"].astype(np.float32)
            lim = limits_of(arr, "physical") if physical else limits_of(arr, "planning", 0.95)
            cache = GeometryCache(M, extract_timeopt_spans(geo, tc0), "tm1")
            base = full_certify(P, V, M, c, arr, geo, tc0, lo, hi, cache, lim, physical)
            Gm, h, jerk = sampled_constraints(geo, M, lim["velocity"], lim["acceleration"], lo, hi, jlim=lim["jerk"])
            r_opt, info = slp_optimize(geo, tc0.astype(np.float64), Gm, h, jerk=jerk)
            accepted, lam_acc, tries = None, 0.0, []
            lam_lo, lam_hi = 0.0, 1.0
            for step in range(10):
                lam = lam_hi if step == 0 else 0.5 * (lam_lo + lam_hi)
                cand = (tc0.astype(np.float64) + lam * (r_opt - tc0.astype(np.float64))).astype(np.float32)
                if (cand <= 0).any():
                    st = ("INVALID", "INVALID")
                else:
                    st = full_certify(P, V, M, c, arr, geo, cand, lo, hi, cache, lim, physical)
                tries.append({"lam": lam, "K3": st[0], "K4": st[1]})
                if st == ("CERTIFIED", "CERTIFIED"):
                    accepted, lam_acc = cand, lam
                    if step == 0:
                        break
                    lam_lo = lam
                else:
                    lam_hi = lam
            Tn = duration(tc0)
            Ta = duration(accepted) if accepted is not None else None
            row = {"path": f.stem, "npz_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                   "native": {"K3": base[0], "K4": base[1], "duration_bracket_s": Tn},
                   "program": info, "constraints": int(len(h)), "tries": tries,
                   "accepted_lambda": lam_acc if accepted is not None else None,
                   "accepted_timing_float32_hex": [float(x).hex() for x in accepted.astype(np.float64)] if accepted is not None else None,
                   "retimed_duration_bracket_s": Ta,
                   "gain_vs_native_pct": (100 * (Tn[0] - Ta[1]) / Tn[1]) if (Ta and Tn) else None,  # conservative
                   "seconds": time.perf_counter() - t0}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("path", "native", "accepted_lambda", "gain_vs_native_pct", "seconds")}
                             | {"native_feasible_in_program": info["native_feasible_in_sampled_program"]}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "K4_P6_CERTIFIED_RETIMING_EXPLORATORY", "limit_scale": a.limit_scale,
                                 "contract_sha256": c["contract_sha256"],
                                 "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
