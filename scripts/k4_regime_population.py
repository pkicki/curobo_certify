#!/usr/bin/env python3
"""K4 regime search (EXPLORATORY): where does TimeOpt's sampled torque enforcement miss violations?

Knob: TimeOpt's enforcement grid, num_intervals (uniform constraint intervals in the path parameter; cuRobo
default 511 -> 512 nodes, plus path knots and timing breaks). Coarser grids are cheaper; the question is how
much continuous-time violation they let through and whether sampled post-checks catch it.
Planning model: identified + URDF hand + PAYLOAD_SYNTH_3KG_CYL (the torque-binding, corrected deployment model,
k4_model_hook). The num_intervals=511 condition is K4-Panda-CorrectedDev40 / IDENT_HAND_3KG (same pool, model,
margin) and is not re-solved.
  freeze    preregister K4-Panda-RegimeDev40 (no solve): frozen K4-Panda-Dev40 pool, 0.95, conditions
            NI255 NI127 NI063 NI031 NI015; first-K = 40 per condition; failures retained.
  generate  RTX 3090, hook installed per condition with the num_intervals override (audited).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

COHORT = "K4-Panda-RegimeDev40"
K = 40
BASE_CONTRACT = ROOT / "docs/reports/k4/identified/panda_contract_identified_plus_hand.json"
PAYLOAD = "PAYLOAD_SYNTH_3KG_CYL"
CONDITIONS = {"NI255": 255, "NI127": 127, "NI063": 63, "NI031": 31, "NI015": 15}
POP = ROOT / "docs/reports/k4/population" / f"{COHORT}.json"


def target_contract():
    from curobo_certify.k4_model_hook import planning_contract
    return planning_contract(json.loads(BASE_CONTRACT.read_text()), PAYLOAD)


def freeze():
    from curobo_certify.margin_campaign import margin_label
    from curobo_certify.query_pipeline import _canonical_hash, verify_query
    base_f = ROOT / "docs/reports/k4/population/K4-Panda-Dev40.json"
    base = sorted((r for r in json.loads(base_f.read_text()) if r["arm"] == "primary" and r["repeat"] == 0),
                  key=lambda r: r["pool_order"])
    tc = target_contract()
    rows = []
    for cond, ni in CONDITIONS.items():
        for b in base:
            label = margin_label("0.95")
            row = {k: b[k] for k in ("schema", "robot", "scene_id", "scene_stratum", "magnitude_stratum", "seed",
                                     "query_index", "goal_delta_rad", "start_joint_positions_rad",
                                     "goal_joint_positions_rad", "query_resolution", "planner", "timeopt",
                                     "source_geometry_contract", "pool_order")}
            row.update({"protocol": "k4_panda_regime_v1", "cohort": COHORT, "condition": cond, "num_intervals": ni,
                        "payload": PAYLOAD, "planning_model_contract_sha256": tc["contract_sha256"],
                        "arm": "primary", "repeat": 0, "cohort_k_eligible": K,
                        "pair_id": f"k4-reg-{cond.lower()}-{b['pool_order']:04d}", "solver_condition": label,
                        "planning_margin": "0.95", "source_id": f"k4-reg-{cond.lower()}-{b['pool_order']:04d}-{label}",
                        "base_source_id": b["source_id"]})
            row["source_query_sha256"] = _canonical_hash(row)
            if not verify_query(row):
                raise AssertionError("freeze")
            rows.append(row)
    data = (json.dumps(rows, sort_keys=True, indent=2) + "\n").encode()
    if POP.exists() and POP.read_bytes() != data:
        raise SystemExit("REFUSING_TO_OVERWRITE_FROZEN")
    POP.write_bytes(data)
    man = {"cohort": COHORT, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows), "exploratory": True,
           "base_manifest_sha256": hashlib.sha256(base_f.read_bytes()).hexdigest(),
           "planning_model_contract_sha256": tc["contract_sha256"], "conditions": CONDITIONS,
           "reference_condition": "num_intervals=511: K4-Panda-CorrectedDev40/IDENT_HAND_3KG",
           "rule": "first-K=40 per condition on the 0.95 arm over the frozen pool order; failures retained",
           "frozen_before_any_solve": True}
    POP.with_name(f"{COHORT}_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(man["sha256"], len(rows))


def generate(expected_sha, out_dir: Path, only=None):
    raw = POP.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise SystemExit("FROZEN_MANIFEST_SHA_MISMATCH")
    rows = json.loads(raw)
    import torch
    import generate_hardcase_population as GH
    import generate_native_pilot as G
    import k4_generate_population as KG
    from curobo_certify import k4_model_hook as MH
    from curobo_certify.formal_cohorts import first_k_selection
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 3090":
        raise SystemExit("WRONG_GPU")
    compat = G._install_cuda_property_compatibility()
    GH.install_config_observer()
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "generation_summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {"cohort": COHORT, "query_manifest_sha256": expected_sha,
                                                              "gpu": gpu, "conditions": {}}
    tc = target_contract()
    for cond, ni in CONDITIONS.items():
        if only and cond not in only:
            continue
        audit = MH.install_timeopt_model_hook(tc, {"num_intervals": ni})
        prim = sorted((r for r in rows if r["condition"] == cond), key=lambda r: r["pool_order"])
        status = []
        for r in prim:
            n0 = len(audit)
            rec = KG.solve(r, out_dir, gpu, compat)
            if len(audit) > n0 and not (audit[-1]["masses_match"] and audit[-1]["cfg"]["num_intervals"] == ni
                                        and audit[-1]["contract_sha256"] == tc["contract_sha256"]):
                raise SystemExit("HOOK_AUDIT_FAILED " + json.dumps({k: v for k, v in audit[-1].items() if k != "links"}))
            status.append((r["pool_order"], rec.get("planner_success") is True))
            if sum(ok for _, ok in status) >= K:
                break
        MH.uninstall()
        sel = first_k_selection(status, K)
        att = set(sel["attempted"])
        ledger = []
        for r in prim:
            if r["pool_order"] not in att:
                continue
            rec = json.loads((out_dir / f"{r['source_id']}.json").read_text())
            npz = out_dir / f"{r['source_id']}.npz"
            ledger.append({"source_id": r["source_id"], "pair_id": r["pair_id"], "arm": "primary", "repeat": 0,
                           "pool_order": r["pool_order"], "scene": r["scene_id"], "magnitude": r["magnitude_stratum"],
                           "condition": cond, "num_intervals": ni, "payload": PAYLOAD, "base_source_id": r["base_source_id"],
                           "status": rec.get("status"), "planner_success": rec.get("planner_success") is True,
                           "selected_eligible": r["pool_order"] in sel["selected"],
                           "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest() if npz.exists() else None,
                           "reason": rec.get("reason")})
        summary["conditions"][cond] = {"num_intervals": ni, "planning_model_contract_sha256": tc["contract_sha256"],
                                       "first_k": sel, "hook_constructions": len(audit),
                                       "hook_audit_sample": {k: v for k, v in (audit[0] if audit else {}).items() if k != "links"},
                                       "ledger": ledger}
        sp.write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps({"condition": cond, "attempted": len(sel["attempted"]), "selected": len(sel["selected"])}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("freeze", "generate"))
    ap.add_argument("--expected-sha256")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs/k4/regime/native")
    a = ap.parse_args()
    freeze() if a.cmd == "freeze" else generate(a.expected_sha256, a.out_dir, a.only)
