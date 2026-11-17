#!/usr/bin/env python3
"""K4 P0: freeze the EXACT rigid-body model used by pinned cuRobo TimeOpt for Panda.

Builds the planner exactly as the native generator does (franka.yml, pinned c301cb00), takes
the TimeOpt optimizer's own KinematicsParams/Dynamics objects, and exports:
  * every float32 tensor that defines the dynamics (fixed transforms, parent map, joint map/types,
    mimic offsets, masses/CoMs, packed inertias), joint names, gravity, effort limits and the
    planning torque limits actually enforced (limit_scale x effort);
  * native cuRobo GPU inverse dynamics tau = F(q, qd, qdd) on seeded states (gravity-only, random
    full states, coefficient-identity triples), all float32 bit patterns retained.
Nothing is modified or "corrected"; physically suspicious parameters are reported by
k4_model_contract.py, not here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="output .npz (arrays) ; .json sidecar written next to it")
    ap.add_argument("--states", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    import torch
    import generate_native_pilot as G
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    from curobo.content import get_robot_configs_path
    G._install_cuda_property_compatibility()
    cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model=None, graph_planner_algorithm="rrt",
                                  enable_trajopt=True, enable_timeopt=True, enable_inverse_dynamics=True,
                                  use_cuda_graph=True, num_ik_seeds=4, num_trajopt_seeds=1, random_seed=1)
    with MotionPlanner(cfg) as planner:
        # TimeOpt is instantiated lazily by the first plan: run one small warm-up request
        # (HOME -> HOME + 0.1 rad on joint 1). Its output is discarded; only the model is read.
        start = planner.default_joint_state.position.reshape(1, -1).clone()
        goal = start.clone(); goal[0, 0] += 0.1
        planner.plan_cspace(JointState.from_position(goal, joint_names=planner.joint_names),
                            JointState.from_position(start, joint_names=planner.joint_names),
                            max_attempts=2, enable_graph_attempt=0, finetune_attempts=0)
        torch.cuda.synchronize()
        opt = planner.trajopt_solver._timeopt.optimizer
        kin = opt.config.robot.kinematics_config
        dyn = opt._dynamics
        arrays, meta = {}, {}
        for name in ("fixed_transforms", "link_map", "joint_map", "joint_map_type", "joint_offset_map",
                     "link_masses_com", "link_inertias"):
            v = getattr(kin, name, None)
            if v is not None:
                arrays["kin_" + name] = v.detach().cpu().numpy().copy()
        for name in ("link_names", "joint_names", "lock_jnames", "tool_frames", "base_link"):
            v = getattr(kin, name, None)
            if v is not None:
                meta[name] = list(v) if not isinstance(v, str) else v
        lim = kin.joint_limits
        for fam in ("position", "velocity", "acceleration", "jerk", "effort"):
            v = getattr(lim, fam, None)
            if v is not None:
                arrays["limit_" + fam] = v.detach().cpu().numpy().copy()
        arrays["timeopt_limits_scaled"] = np.stack([t.detach().cpu().numpy() for t in opt._limits])
        meta["timeopt_limits_order"] = ["velocity", "acceleration", "jerk", "torque_lower", "torque_upper"]
        meta["gravity"] = list(dyn.config.gravity)
        meta["limit_scale"] = opt.config.limit_scale
        meta["num_dof"] = int(kin.num_dof)
        meta["dynamics_backend_joint_names"] = list(getattr(dyn, "_backend_joint_names", []) or [])
        # ---- native GPU inverse dynamics on seeded states (float32) ----
        rng = np.random.default_rng(a.seed)
        n, d = a.states, int(kin.num_dof)
        plo, phi = arrays["limit_position"][0], arrays["limit_position"][1]
        q = (plo + (phi - plo) * rng.random((n, d))).astype(np.float32)
        qd = rng.uniform(-2.0, 2.0, (n, d)).astype(np.float32)
        qdd = rng.uniform(-10.0, 10.0, (n, d)).astype(np.float32)
        zero = np.zeros_like(q)
        sets = {"gravity_only": (q, zero, zero), "full": (q, qd, qdd),
                "B_probe": (q, zero, qd), "A_probe": (q, qd, qdd)}
        dyn.setup_batch_size(batch_size=n)
        names = meta["dynamics_backend_joint_names"] or None
        for key, (qq, vv, aa) in sets.items():
            js = JointState(position=torch.as_tensor(qq, device=opt.device), velocity=torch.as_tensor(vv, device=opt.device),
                            acceleration=torch.as_tensor(aa, device=opt.device), joint_names=names)
            tau = dyn.compute_inverse_dynamics(js)
            torch.cuda.synchronize()
            arrays[f"ids_{key}_q"], arrays[f"ids_{key}_qd"], arrays[f"ids_{key}_qdd"] = qq, vv, aa
            arrays[f"ids_{key}_tau"] = tau.detach().cpu().numpy().astype(np.float32).copy()
        meta["robot_config_sha256"] = hashlib.sha256((Path(get_robot_configs_path()) / "franka.yml").read_bytes()).hexdigest()
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["seed"] = a.seed
        meta["states"] = n
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **arrays)
    meta["arrays_sha256"] = hashlib.sha256(a.out.read_bytes()).hexdigest()
    meta["array_shapes"] = {k: list(v.shape) for k, v in arrays.items()}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta.get(k) for k in ("num_dof", "gravity", "limit_scale", "link_names", "joint_names")}, indent=1))
    print({k: v.tolist() for k, v in arrays.items() if k.startswith("limit_effort") or k == "timeopt_limits_scaled"})


if __name__ == "__main__":
    main()
