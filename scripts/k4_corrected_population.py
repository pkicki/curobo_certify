#!/usr/bin/env python3
"""K4 'corrected model' study: cuRobo TimeOpt planning WITH the identified Panda model (k4_model_hook).

  freeze    preregister K4-Panda-CorrectedDev40 (no solve): the frozen K4-Panda-Dev40 pool (primary repeat-0
            rows, frozen pool order), arm 0.95 only, three conditions
              IDENT_HAND            identified + URDF hand contract (9fe7f2dc...)
              IDENT_HAND_3KG        + PAYLOAD_SYNTH_3KG_CYL (hypothetical, rated)
              IDENT_HAND_5KG        + PAYLOAD_SYNTH_5KG_CYL (hypothetical, above rated)
            first-K = 40 successes per condition over the frozen 70-query pool; failures retained.
  generate  solve on the RTX 3090 with the model hook installed per condition (audited read-back, gravity
            9.80665); per-row worker = k4_generate_population.solve (frozen K3 worker inside).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

COHORT = "K4-Panda-CorrectedDev40"
K = 40
BASE_CONTRACT = ROOT / "docs/reports/k4/identified/panda_contract_identified_plus_hand.json"
CONDITIONS = {"IDENT_HAND": None, "IDENT_HAND_3KG": "PAYLOAD_SYNTH_3KG_CYL", "IDENT_HAND_5KG": "PAYLOAD_SYNTH_5KG_CYL"}
TAGS = {"IDENT_HAND": "c0", "IDENT_HAND_3KG": "c3", "IDENT_HAND_5KG": "c5"}
POP = ROOT / "docs/reports/k4/population" / f"{COHORT}.json"


def target_contract(cond):
    from curobo_certify.k4_model_hook import planning_contract
    return planning_contract(json.loads(BASE_CONTRACT.read_text()), CONDITIONS[cond])


def freeze():
    from curobo_certify.margin_campaign import margin_label
    from curobo_certify.query_pipeline import _canonical_hash, verify_query
    base_f = ROOT / "docs/reports/k4/population/K4-Panda-Dev40.json"
    base = sorted((r for r in json.loads(base_f.read_text()) if r["arm"] == "primary" and r["repeat"] == 0),
                  key=lambda r: r["pool_order"])
    rows = []
    for cond, pay in CONDITIONS.items():
        tc = target_contract(cond)
        for b in base:
            label = margin_label("0.95")
            row = {k: b[k] for k in ("schema", "robot", "scene_id", "scene_stratum", "magnitude_stratum", "seed",
                                     "query_index", "goal_delta_rad", "start_joint_positions_rad",
                                     "goal_joint_positions_rad", "query_resolution", "planner", "timeopt",
                                     "source_geometry_contract", "pool_order")}
            row.update({"protocol": "k4_panda_corrected_model_v1", "cohort": COHORT, "condition": cond,
                        "payload": pay, "planning_model_contract_sha256": tc["contract_sha256"],
                        "arm": "primary", "repeat": 0, "cohort_k_eligible": K,
                        "pair_id": f"k4-corr-{TAGS[cond]}-{b['pool_order']:04d}", "solver_condition": label,
                        "planning_margin": "0.95", "source_id": f"k4-corr-{TAGS[cond]}-{b['pool_order']:04d}-{label}",
                        "base_source_id": b["source_id"]})
            row["source_query_sha256"] = _canonical_hash(row)
            if not verify_query(row):
                raise AssertionError("freeze")
            rows.append(row)
    data = (json.dumps(rows, sort_keys=True, indent=2) + "\n").encode()
    if POP.exists() and POP.read_bytes() != data:
        raise SystemExit("REFUSING_TO_OVERWRITE_FROZEN")
    POP.write_bytes(data)
    man = {"cohort": COHORT, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows),
           "base_manifest_sha256": hashlib.sha256(base_f.read_bytes()).hexdigest(),
           "base_planning_contract_sha256": json.loads(BASE_CONTRACT.read_text())["contract_sha256"],
           "conditions": {c: {"payload": p, "planning_model_contract_sha256": target_contract(c)["contract_sha256"]}
                          for c, p in CONDITIONS.items()},
           "rule": "first-K=40 per condition on the 0.95 arm over the frozen pool order; failures retained; native "
                   "TimeOpt plans WITH the identified+hand model (k4_model_hook, audited); 0.99 arm not registered",
           "frozen_before_any_solve": True}
    POP.with_name(f"{COHORT}_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(man["sha256"], len(rows))


def generate(expected_sha, out_dir: Path):
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
    summary = {"cohort": COHORT, "query_manifest_sha256": expected_sha, "gpu": gpu, "conditions": {}}
    for cond in CONDITIONS:
        tc = target_contract(cond)
        audit = MH.install_timeopt_model_hook(tc)
        prim = sorted((r for r in rows if r["condition"] == cond), key=lambda r: r["pool_order"])
        status = []
        for r in prim:
            if r["planning_model_contract_sha256"] != tc["contract_sha256"]:
                raise SystemExit("CONTRACT_MISMATCH")
            n0 = len(audit)
            rec = KG.solve(r, out_dir, gpu, compat)
            if len(audit) > n0 and not (audit[-1]["masses_match"] and audit[-1]["contract_sha256"] == tc["contract_sha256"]
                                        and abs(audit[-1]["gravity"][2] + 9.80665) < 1e-9):
                raise SystemExit("MODEL_HOOK_AUDIT_FAILED " + json.dumps(audit[-1])[:400])
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
                           "condition": cond, "payload": r["payload"], "base_source_id": r["base_source_id"],
                           "status": rec.get("status"), "planner_success": rec.get("planner_success") is True,
                           "selected_eligible": r["pool_order"] in sel["selected"],
                           "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest() if npz.exists() else None,
                           "reason": rec.get("reason")})
        summary["conditions"][cond] = {"planning_model_contract_sha256": tc["contract_sha256"], "first_k": sel,
                                       "hook_constructions": len(audit),
                                       "hook_audit_sample": {k: v for k, v in (audit[0] if audit else {}).items() if k != "links"},
                                       "ledger": ledger}
        print(json.dumps({"condition": cond, "attempted": len(sel["attempted"]), "selected": len(sel["selected"])}), flush=True)
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("freeze", "generate"))
    ap.add_argument("--expected-sha256")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs/k4/corrected/native")
    a = ap.parse_args()
    freeze() if a.cmd == "freeze" else generate(a.expected_sha256, a.out_dir)
