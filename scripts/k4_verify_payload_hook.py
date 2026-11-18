#!/usr/bin/env python3
"""Verify that the TimeOpt payload hook really changes cuRobo TimeOpt's dynamics (GPU, before generation).

For each payload: build the planner as the generator does, install the hook, run one warm-up plan (TimeOpt is
constructed lazily), then evaluate TimeOpt's own Dynamics at seeded states and compare with this repository's
interval RNEA on (a) contract v2 + payload and (b) contract v2 without payload. PASS requires (a) to agree within
1e-4 relative (float32 native) and (b) to differ by more than 1 N m somewhere (the payload is actually present).
"""
import json
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    import torch
    import generate_native_pilot as G
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    from curobo_certify import k4_payload as KP
    from curobo_certify.k4_interval import IA
    from curobo_certify.k4_panda_model import PandaModel, load_contract
    G._install_cuda_property_compatibility()
    v2 = load_contract(ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
    out = {}
    rng = np.random.default_rng(11)
    q = rng.uniform(-2.0, 2.0, (64, 7)).astype(np.float32)
    qd = rng.uniform(-1.5, 1.5, (64, 7)).astype(np.float32)
    qdd = rng.uniform(-5, 5, (64, 7)).astype(np.float32)
    for name in KP.PAYLOADS:
        audit = KP.install_timeopt_payload_hook(name)
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
        box = lambda a: [IA(a[:, j].astype(np.float64)) for j in range(7)]  # noqa: E731
        ref_p = np.stack([t.mid() for t in PandaModel(KP.with_payload(v2, name)).torque(box(q), box(qd), box(qdd))], 1)
        ref_0 = np.stack([t.mid() for t in PandaModel(v2).torque(box(q), box(qd), box(qdd))], 1)
        rel = float((np.abs(tn - ref_p) / np.maximum(1, np.abs(ref_p))).max())
        diff0 = float(np.abs(tn - ref_0).max())
        out[name] = {"hook_audit": audit, "max_rel_native_vs_payload_contract": rel,
                     "max_abs_native_vs_no_payload_contract": diff0,
                     "pass": bool(rel < 1e-4 and diff0 > 1.0)}
        print(name, out[name]["pass"], rel, diff0, flush=True)
    p = ROOT / "docs/reports/k4/payload/hook_verification.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    if not all(v["pass"] for v in out.values()):
        raise SystemExit("PAYLOAD_HOOK_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
