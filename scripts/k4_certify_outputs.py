#!/usr/bin/env python3
"""K4 P4 per-output certification (CPU): MODEL_K3, MODEL_K4, combined, planning-contract K4.

For each frozen native NPZ:
  MODEL_K3   compiled K3 producer + independent verifier (frozen K3 code, physical limits, adaptive
             depth 12 / 8192 nodes) - unchanged K3 machinery, not modified.
  MODEL_K4   tm1 geometry cache certificate (physical effort limits of contract v2) + independent
             world-frame verifier (verify_k4, refine 8) or witness replay (verify_k4_witness);
             a certificate counts only if the verifier passes.
  PLANNING_K4  the same against cuRobo TimeOpt's own scaled torque limits (0.95 x effort, float32).
  K3_PLUS_K4 CERTIFIED iff both MODEL_K3 and MODEL_K4 are verified CERTIFIED on the identical bytes.
  dense_point_max_util: rigorous point torques at 24 points/span (sampled diagnostic of conservatism).
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
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402
from curobo_certify.k4_reuse import GeometryCache  # noqa: E402
from curobo_certify.k4_verify import verify_k4, verify_k4_witness  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402
from bench_certification_methods import limits_of  # noqa: E402
from k4_p3_compare import dense_max_util  # noqa: E402


def k4(M, c, spans, lo, hi, physical):
    t0 = time.perf_counter()
    C = GeometryCache(M, spans, "tm1")
    r = C.certify(spans, lo, hi)
    t1 = time.perf_counter()
    if r.status == "CERTIFIED":
        v = verify_k4(spans, c, r, lo, hi, expect_physical=physical, refine=8)
    elif r.status == "VIOLATED_TORQUE":
        v = verify_k4_witness(spans, c, r, lo, hi)
    else:
        v = {"pass": None, "reason": "nothing to verify"}
    t2 = time.perf_counter()
    final = r.status
    if r.status in ("CERTIFIED", "VIOLATED_TORQUE") and not v["pass"]:
        final = "REJECTED_BY_VERIFIER"
    return {"status": final, "producer": r.status, "verifier": v, "nodes": r.nodes, "depth": r.max_depth,
            "leaves": len(r.leaves), "bound_util": r.worst_utilization, "witness": r.witness,
            "seconds_producer_incl_prep": t1 - t0, "seconds_prep": C.prep_seconds, "seconds_verifier": t2 - t1}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--contract", type=Path, default=ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = load_contract(a.contract)
    M = PandaModel(c)
    lo = np.array([float(Q(v)) for v in c["model"]["effort_lower"]])
    hi = np.array([float(Q(v)) for v in c["model"]["effort_upper"]])
    plo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
    phi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
    rows = []
    with tempfile.TemporaryDirectory() as d:
        builds = [K.build("k3_certifier.cpp", Path(d) / "c.so"), K.build("k3_verifier.cpp", Path(d) / "v.so")]
        P, V = K.Producer(Path(d) / "c.so"), K.Verifier(Path(d) / "v.so")
        for f in a.npz:
            raw = f.read_bytes()
            with np.load(f) as z:
                arr = {k: z[k].copy() for k in z.files}
            # ---- MODEL_K3 (frozen compiled K3 producer + verifier, physical limits) ----
            t0 = time.perf_counter()
            args = (arr["effective_geometry"], arr["timing_controls"], arr["requested_start"][0], arr["requested_goal"][0],
                    limits_of(arr, "physical"))
            r3 = P.certify(*args, max_depth=12, max_nodes=8192)
            k3 = r3["status"]
            if k3 == "CERTIFIED":
                vv = V.verify(*args, r3["leaves"], K.input_digest(*args, None))
                k3 = "CERTIFIED" if vv["pass"] else "REJECTED_BY_VERIFIER"
            t3 = time.perf_counter() - t0
            spans = extract_timeopt_spans(arr["effective_geometry"], arr["timing_controls"])
            phys = k4(M, c, spans, lo, hi, True)
            plan = k4(M, c, spans, plo, phi, False)
            pm = dense_max_util(M, spans, lo, hi)
            row = {"path": f.stem, "npz_sha256": hashlib.sha256(raw).hexdigest(),
                   "native_duration_s": float(arr["native_duration"][0]),
                   "MODEL_K3": k3, "MODEL_K3_seconds": t3, "MODEL_K4": phys["status"], "K4_physical": phys,
                   "PLANNING_K4": plan["status"], "K4_planning": plan,
                   "K3_PLUS_K4": ("CERTIFIED" if k3 == "CERTIFIED" and phys["status"] == "CERTIFIED"
                                  else "VIOLATED" if k3 in ("VIOLATED", "VIOLATED_BOUNDARY") or phys["status"] in ("VIOLATED_TORQUE", "VIOLATED_X")
                                  else "UNKNOWN"),
                   "dense_point_max_util_physical": float(pm.max()), "dense_point_max_util_per_joint": pm.tolist()}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("path", "MODEL_K3", "MODEL_K4", "PLANNING_K4", "K3_PLUS_K4",
                                                   "dense_point_max_util_physical")}), flush=True)
    out = {"scope": "K4_P4_PER_OUTPUT_CERTIFICATION", "contract_sha256": c["contract_sha256"], "k3_builds": builds,
           "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
