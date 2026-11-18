#!/usr/bin/env python3
"""Escalate verifier non-confirmations (REJECTED_BY_VERIFIER at protocol refinement 8) per the preregistration:
re-run BOTH independent verifiers (Python world-frame verify_k4, compiled C++ k4v_verify) at refinement 12 on the same
bytes, contract and limits. Output {source_id|column: {python_refine12, compiled_refine12, ...}} for k4_analyze_confirm."""
import argparse
import glob
import json
import sys
import tempfile
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=ROOT / "runs/k4/confirm")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from curobo_certify import k4_compiled as KC
    from curobo_certify.k4_panda_model import PandaModel
    from curobo_certify.k4_payload import with_payload
    from curobo_certify.k4_reuse import GeometryCache
    from curobo_certify.k4_verify import verify_k4
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    out = {}
    tmp = tempfile.TemporaryDirectory()
    KC.build(Path(tmp.name) / "p.so"); KC.build_verifier(Path(tmp.name) / "v.so")
    for f in sorted(glob.glob(str(a.dir / "cert/part_*.json"))):
        d = json.loads(Path(f).read_text())
        for r in d["rows"]:
            for col, key, physical in (("MODEL_K4", "K4_physical", True), ("PLANNING_K4", "K4_planning", False)):
                if r[col] != "REJECTED_BY_VERIFIER":
                    continue
                cond = {"ca": "CA", "cb": "CB", "cc": "CC", "cd": "CD"}[r["path"].split("-")[2]]
                base = json.loads((ROOT / ("docs/reports/k4/p0/panda_model_contract_v2.json" if cond in ("CA", "CB")
                                           else "docs/reports/k4/identified/panda_contract_identified_plus_hand.json")).read_text())
                if "ident" in Path(f).name:
                    base = json.loads((ROOT / "docs/reports/k4/identified/panda_contract_identified_plus_hand.json").read_text())
                c = with_payload(base, d["payload"]) if d.get("payload") else base
                M = PandaModel(c)
                if physical:
                    lo = np.array([float(Q(v)) for v in c["model"]["effort_lower"]]); hi = -lo
                else:
                    lo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
                    hi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
                z = np.load(a.dir / "native" / f"{r['path']}.npz")
                g, tc = z["effective_geometry"], z["timing_controls"]
                sp = extract_timeopt_spans(g, tc)
                rp = GeometryCache(M, sp, "tm1").certify(sp, lo, hi)
                pv = verify_k4(sp, c, rp, lo, hi, expect_physical=physical, refine=12, expected_contract_sha256=c["contract_sha256"])
                P = KC.CompiledPath(M, Path(tmp.name) / "p.so")
                V = KC.CompiledVerifier(c, Path(tmp.name) / "v.so", expected_contract_sha256=c["contract_sha256"])
                rc = P.certify(g, tc, lo, hi, threads=8)
                cv = V.verify(g, tc, rc, lo, hi, refine=12, threads=8) if rc["status"] == "CERTIFIED" else {"pass": False, "reason": rc["status"]}
                out[f"{r['path']}|{col}"] = {"file": Path(f).name, "python_producer": rp.status, "python_refine12": bool(pv["pass"]),
                                             "compiled_producer": rc["status"], "compiled_refine12": bool(cv["pass"]),
                                             "python_reason": pv.get("reason"), "compiled_reason": cv.get("reason")}
                print(r["path"], col, out[f"{r['path']}|{col}"], flush=True)
    (a.out or a.dir / "escalation.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
