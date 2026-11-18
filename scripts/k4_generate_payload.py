#!/usr/bin/env python3
"""Solve the preregistered hypothetical-payload K4 population (RTX 3090), payload by payload.

For each payload: install the TimeOpt payload hook (k4_payload.install_timeopt_payload_hook), solve the 0.95
arm in frozen pool order until the K-th success (first-K), then the 0.99 arm for exactly the attempted source
queries. Every attempt retained; per-row worker = k4_generate_population.solve (frozen K3 worker inside).
The hook's read-back audit is stored per payload; generation refuses to continue if it does not show the
intended payload in TimeOpt's kinematics.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import generate_hardcase_population as GH  # noqa: E402
import generate_native_pilot as G  # noqa: E402
import k4_generate_population as KG  # noqa: E402
from curobo_certify import k4_payload as KP  # noqa: E402
from curobo_certify.formal_cohorts import first_k_selection  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    raw = a.queries.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.expected_sha256:
        raise SystemExit("FROZEN_MANIFEST_SHA_MISMATCH")
    rows = json.loads(raw)
    import torch
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 3090":
        raise SystemExit("WRONG_GPU")
    compat = G._install_cuda_property_compatibility()
    GH.install_config_observer()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"cohort": KP.COHORT, "query_manifest_sha256": a.expected_sha256, "gpu": gpu, "payloads": {}}
    for name in KP.PAYLOADS:
        audit = KP.install_timeopt_payload_hook(name)
        R = [r for r in rows if r["payload"] == name]
        prim = sorted((r for r in R if r["arm"] == "primary"), key=lambda r: r["pool_order"])
        status = []
        for r in prim:
            rec = KG.solve(r, a.out_dir, gpu, compat)
            if audit:
                exp = KP.inertial(name)
                got = audit[-1]
                if abs(got["masses_com"][3] - float(exp["mass"])) > 1e-6:
                    raise SystemExit("PAYLOAD_HOOK_AUDIT_FAILED " + json.dumps(got))
            status.append((r["pool_order"], rec.get("planner_success") is True))
            if sum(ok for _, ok in status) >= KP.K:
                break
        sel = first_k_selection(status, KP.K)
        pairs = {prim[o]["pair_id"] for o in sel["attempted"]}
        for r in sorted((r for r in R if r["arm"] == "secondary" and r["pair_id"] in pairs), key=lambda r: r["pool_order"]):
            KG.solve(r, a.out_dir, gpu, compat)
        ledger = []
        for r in R:
            if r["pair_id"] not in pairs:
                continue
            rec = json.loads((a.out_dir / f"{r['source_id']}.json").read_text())
            npz = a.out_dir / f"{r['source_id']}.npz"
            ledger.append({"source_id": r["source_id"], "pair_id": r["pair_id"], "arm": r["arm"], "repeat": 0,
                           "pool_order": r["pool_order"], "scene": r["scene_id"], "magnitude": r["magnitude_stratum"],
                           "payload": name, "status": rec.get("status"), "planner_success": rec.get("planner_success") is True,
                           "selected_eligible": r["arm"] == "primary" and r["pool_order"] in sel["selected"],
                           "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest() if npz.exists() else None,
                           "reason": rec.get("reason")})
        summary["payloads"][name] = {"first_k": sel, "hook_audit_samples": audit[:2], "hook_constructions": len(audit),
                                     "ledger": ledger}
        print(json.dumps({"payload": name, "attempted": len(sel["attempted"]), "selected": len(sel["selected"])}), flush=True)
    (a.out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
