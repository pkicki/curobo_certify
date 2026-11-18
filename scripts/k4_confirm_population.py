#!/usr/bin/env python3
"""K4-Panda-Confirm150: preregistered confirmation cohort (fresh queries; frozen BEFORE any solve).

Fresh pool: same generator distribution as K4-Panda-Dev40 (4 scenes x 3 motion magnitudes, planner HOME start), new
SHA-256-derived seed, leakage audit against every prior Panda query file (K3 formal pools, K4 Dev40 pool).
POOL = 230 ordered queries; first-K = 150 successes PER CONDITION over the frozen order (failures retained).
Conditions (all 0.95 TimeOpt margin, arm 'primary'):
  CA_CUROBO          cuRobo model, cuRobo default payload            (deployment default)
  CB_CUROBO_3KG      cuRobo model + PAYLOAD_SYNTH_3KG_CYL (hook)      (model-mismatch exposure)
  CC_CORR_3KG        identified+hand + 3 kg (k4_model_hook)          (corrected planning)
  CD_CORR_3KG_NI063  CC with TimeOpt num_intervals = 63               (coarse enforcement grid)
The analysis plan and decision rules are in docs/reports/k4/confirm/PREREGISTRATION.md (committed with this file).
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

COHORT = "K4-Panda-Confirm150"
PROTOCOL = "k4_panda_confirm_v1"
SALT = "curobo_certify/k4_panda_confirm_v1/"
K, POOL = 150, 230
IH = ROOT / "docs/reports/k4/identified/panda_contract_identified_plus_hand.json"
CONDITIONS = {"CA_CUROBO": (None, None, None), "CB_CUROBO_3KG": (None, "PAYLOAD_SYNTH_3KG_CYL", None),
              "CC_CORR_3KG": ("IH", "PAYLOAD_SYNTH_3KG_CYL", None), "CD_CORR_3KG_NI063": ("IH", "PAYLOAD_SYNTH_3KG_CYL", 63)}
TAG = {"CA_CUROBO": "ca", "CB_CUROBO_3KG": "cb", "CC_CORR_3KG": "cc", "CD_CORR_3KG_NI063": "cd"}
POP = ROOT / "docs/reports/k4/population" / f"{COHORT}.json"
PRIOR = {
    "pilot_seed20261110": "runs/native_pilot_queries.json",
    "exploratory_seed20261111": "runs/compiled_hull_panda/exploratory_queries_seed20261111.json",
    "hardcase_seed20261116": "runs/hardcase_seed20261116/queries.json",
    "margins_seed20261117": "runs/margins/queries.json",
    "k3_formal_Panda-Dev40": "docs/reports/formal_panda/queries/Panda-Dev40.json",
    "k3_formal_Panda-Holdout100": "docs/reports/formal_panda/queries/Panda-Holdout100.json",
    "k3_formal_Panda-Confirm300": "docs/reports/formal_panda/queries/Panda-Confirm300.json",
    "k4_K4-Panda-Dev40": "docs/reports/k4/population/K4-Panda-Dev40.json",
}


def seed():
    return int.from_bytes(hashlib.sha256((SALT + COHORT).encode()).digest()[:4], "big") >> 2


def planning(cond):
    from curobo_certify.k4_model_hook import planning_contract
    model, pay, ni = CONDITIONS[cond]
    if model is None:
        return None, pay, ni
    return planning_contract(json.loads(IH.read_text()), pay), pay, ni


def freeze():
    from curobo_certify.formal_cohorts import CELLS, _goal, leakage_audit
    from curobo_certify.margin_campaign import HOME, margin_label
    from curobo_certify.query_pipeline import _canonical_hash, verify_query
    sd = seed()
    rng = random.Random(sd)
    goals = []
    while len(goals) < POOL:
        block = list(CELLS)
        rng.shuffle(block)
        for scene, mag in block:
            if len(goals) < POOL:
                goals.append((scene, mag, _goal(rng, mag)))
    rows = []
    label = margin_label("0.95")
    for cond in CONDITIONS:
        tc, pay, ni = planning(cond)
        for order, (scene, mag, goal) in enumerate(goals):
            pair = f"k4-conf-{order:04d}"
            row = {"schema": "timeopt_native_source_query_v1", "protocol": PROTOCOL, "cohort": COHORT, "condition": cond,
                   "arm": "primary", "repeat": 0, "pool_order": order, "cohort_k_eligible": K,
                   "payload": pay or "PAYLOAD_DEFAULT_0P01KG", "num_intervals": ni or 511,
                   "planning_model_contract_sha256": tc["contract_sha256"] if tc else "CUROBO_NATIVE_MODEL",
                   "robot": "panda", "scene_id": scene, "scene_stratum": scene, "magnitude_stratum": mag, "seed": sd,
                   "query_index": order, "pair_id": pair, "solver_condition": label, "planning_margin": "0.95",
                   "source_id": f"k4-conf-{TAG[cond]}-{order:04d}-{label}",
                   "goal_delta_rad": [g - h for g, h in zip(goal, HOME)], "start_joint_positions_rad": list(HOME),
                   "goal_joint_positions_rad": goal, "query_resolution": "FROZEN_EXPLICIT_CSPACE",
                   "planner": "pinned_curobo_motion_planner.plan_cspace", "timeopt": "pinned_curobo.TrajectoryTimeOpt",
                   "source_geometry_contract": ("selected planner-integrated TimeOpt output: cloned effective cubic "
                                                "geometry, unchanged 32 timing controls, physical actuator limits")}
            row["source_query_sha256"] = _canonical_hash(row)
            if not verify_query(row):
                raise AssertionError("freeze")
            rows.append(row)
    prior = {}
    for n, rel in PRIOR.items():
        rr = json.loads((ROOT / rel).read_text())
        prior[n] = [r for r in rr if r.get("robot") == "panda" and r.get("goal_joint_positions_rad")]
    leak = leakage_audit({COHORT: [r for r in rows if r["condition"] == "CA_CUROBO"]}, prior)
    if not leak["pass"]:
        raise SystemExit("LEAKAGE " + json.dumps(leak)[:500])
    data = (json.dumps(rows, sort_keys=True, indent=2) + "\n").encode()
    if POP.exists() and POP.read_bytes() != data:
        raise SystemExit("REFUSING_TO_OVERWRITE_FROZEN")
    POP.write_bytes(data)
    man = {"cohort": COHORT, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows), "seed": sd, "pool": POOL,
           "k_eligible_per_condition": K,
           "conditions": {c: {"planning_model_contract_sha256": (planning(c)[0] or {}).get("contract_sha256", "CUROBO_NATIVE_MODEL"),
                              "payload": CONDITIONS[c][1], "num_intervals": CONDITIONS[c][2] or 511} for c in CONDITIONS},
           "leakage_audit": leak, "prior_files_sha256": {n: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for n, p in PRIOR.items()},
           "frozen_before_any_solve": True, "preregistration": "docs/reports/k4/confirm/PREREGISTRATION.md"}
    POP.with_name(f"{COHORT}_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(man["sha256"], len(rows), "leakage pass", leak["pass"])


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
    from curobo_certify import k4_payload as KP
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
    for cond in CONDITIONS:
        if (only and cond not in only) or cond in summary["conditions"]:
            continue
        tc, pay, ni = planning(cond)
        MH.uninstall()
        if tc is not None:
            audit = MH.install_timeopt_model_hook(tc, {"num_intervals": ni} if ni else None)
        elif pay is not None:
            audit = KP.install_timeopt_payload_hook(pay)
        else:
            audit = None
        prim = sorted((r for r in rows if r["condition"] == cond), key=lambda r: r["pool_order"])
        status = []
        for r in prim:
            n0 = len(audit) if audit is not None else 0
            rec = KG.solve(r, out_dir, gpu, compat)
            if audit is not None and len(audit) > n0:
                au = audit[-1]
                if tc is not None and not (au["masses_match"] and au["contract_sha256"] == tc["contract_sha256"]
                                           and au["cfg"]["num_intervals"] == (ni or 511)):
                    raise SystemExit("MODEL_HOOK_AUDIT_FAILED")
                if tc is None and abs(au["masses_com"][3] - float(KP.inertial(pay)["mass"])) > 1e-6:
                    raise SystemExit("PAYLOAD_HOOK_AUDIT_FAILED")
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
                           "condition": cond, "payload": r["payload"], "num_intervals": r["num_intervals"],
                           "status": rec.get("status"), "planner_success": rec.get("planner_success") is True,
                           "selected_eligible": r["pool_order"] in sel["selected"],
                           "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest() if npz.exists() else None,
                           "reason": rec.get("reason")})
        summary["conditions"][cond] = {"first_k": sel, "planning_model_contract_sha256": tc["contract_sha256"] if tc else None,
                                       "payload": pay, "num_intervals": ni or 511, "ledger": ledger,
                                       "hook_audit_sample": ({k: v for k, v in audit[0].items() if k != "links"} if audit else None)}
        sp.write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps({"condition": cond, "attempted": len(sel["attempted"]), "selected": len(sel["selected"])}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("freeze", "generate"))
    ap.add_argument("--expected-sha256")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs/k4/confirm/native")
    a = ap.parse_args()
    freeze() if a.cmd == "freeze" else generate(a.expected_sha256, a.out_dir, a.only)
