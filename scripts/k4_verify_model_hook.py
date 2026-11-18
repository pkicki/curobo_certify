#!/usr/bin/env python3
"""Verify that cuRobo TimeOpt really plans with a K4 model contract after k4_model_hook (GPU).

For each condition (identified+hand base contract, optionally + hypothetical payload): install the hook, build
the planner as the generator does, run one warm-up plan, then evaluate TimeOpt's own Dynamics at seeded states
and compare with this repository's interval RNEA on (a) the target contract and (b) cuRobo's original model
(contract v2 [+ same payload]). PASS requires (a) within 1e-4 relative (float32 native) and (b) a difference
> 0.5 N m somewhere (the model really changed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, default=ROOT / "docs/reports/k4/identified/panda_contract_identified_plus_hand.json")
    ap.add_argument("--payloads", nargs="*", default=["none", "PAYLOAD_SYNTH_3KG_CYL", "PAYLOAD_SYNTH_5KG_CYL"])
    ap.add_argument("--out", type=Path, default=ROOT / "docs/reports/k4/corrected/model_hook_verification.json")
    a = ap.parse_args()
    import torch
    import generate_native_pilot as G
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    from curobo_certify import k4_model_hook as MH
    from curobo_certify.k4_interval import IA
    from curobo_certify.k4_panda_model import PandaModel
    from curobo_certify.k4_payload import with_payload
    from curobo_certify.k4_verify import _contract_ok
    G._install_cuda_property_compatibility()
    base = json.loads(a.contract.read_text())
    if not _contract_ok(base, base["contract_sha256"]):
        raise SystemExit("contract hash mismatch")
    v2 = json.loads((ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json").read_text())
    rng = np.random.default_rng(13)
    q = rng.uniform(-2.0, 2.0, (64, 7)).astype(np.float32)
    qd = rng.uniform(-1.5, 1.5, (64, 7)).astype(np.float32)
    qdd = rng.uniform(-5, 5, (64, 7)).astype(np.float32)
    box = lambda x: [IA(x[:, j].astype(np.float64)) for j in range(7)]  # noqa: E731
    out = {"base_contract_sha256": base["contract_sha256"], "conditions": {}}
    for p in a.payloads:
        pay = None if p == "none" else p
        tgt = MH.planning_contract(base, pay)
        ref0 = v2 if pay is None else with_payload(v2, pay)
        audit = MH.install_timeopt_model_hook(tgt)
        cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model=None, graph_planner_algorithm="rrt",
                                      enable_trajopt=True, enable_timeopt=True, enable_inverse_dynamics=True,
                                      use_cuda_graph=True, num_ik_seeds=4, num_trajopt_seeds=1, random_seed=1)
        with MotionPlanner(cfg) as planner:
            s0 = planner.default_joint_state.position.reshape(1, -1).clone()
            g0 = s0.clone(); g0[0, 0] += 0.1
            planner.plan_cspace(JointState.from_position(g0, joint_names=planner.joint_names),
                                JointState.from_position(s0, joint_names=planner.joint_names),
                                max_attempts=2, enable_graph_attempt=0, finetune_attempts=0)
            dyn = planner.trajopt_solver._timeopt.optimizer._dynamics
            dyn.setup_batch_size(batch_size=64)
            dev = planner.default_joint_state.position.device
            js = JointState(position=torch.as_tensor(q, device=dev), velocity=torch.as_tensor(qd, device=dev),
                            acceleration=torch.as_tensor(qdd, device=dev))
            tn = dyn.compute_inverse_dynamics(js).detach().cpu().numpy().astype(np.float64)
        MH.uninstall()
        rt = np.stack([t.mid() for t in PandaModel(tgt).torque(box(q), box(qd), box(qdd))], 1)
        r0 = np.stack([t.mid() for t in PandaModel(ref0).torque(box(q), box(qd), box(qdd))], 1)
        rel = float((np.abs(tn - rt) / np.maximum(1, np.abs(rt))).max())
        d0 = float(np.abs(tn - r0).max())
        rec = {"target_contract_sha256": tgt["contract_sha256"], "hook_constructions": len(audit),
               "audit_gravity": audit[-1]["gravity"] if audit else None,
               "audit_masses_match": bool(audit and audit[-1]["masses_match"]),
               "max_rel_native_vs_target_contract": rel, "max_abs_native_vs_curobo_model": d0,
               "pass": bool(audit and audit[-1]["masses_match"] and rel < 1e-4 and d0 > 0.5)}
        out["conditions"][p] = rec
        print(p, json.dumps(rec), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    if not all(v["pass"] for v in out["conditions"].values()):
        raise SystemExit("MODEL_HOOK_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
