#!/usr/bin/env python3
"""K4 diagnostic: cuRobo NATIVE float32 torque at native production samples (SAMPLED_DIAGNOSTIC).

For each frozen output: the production sampler (cuRobo trajectory_timeopt.sample, as in T1 v2) at
  * the 512 native time-grid times (TimeOpt's output grid),
  * the saved S8193 uniform-time diagnostic times,
  * the saved event-aware S32768 diagnostic times,
then cuRobo's own GPU inverse dynamics (TimeOpt optimizer's Dynamics object) on (q, v, a). Reports
max |tau| / limit per joint against physical effort and TimeOpt planning (0.95 x effort) limits.
Never a proof: a sampled maximum below a limit says nothing between samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--payload", default=None, help="install the hypothetical-payload TimeOpt hook (k4_payload)")
    ap.add_argument("--model-contract", type=Path, default=None,
                    help="plan/evaluate with this model contract (k4_model_hook; combined with --payload if given)")
    a = ap.parse_args()
    import torch
    import generate_native_pilot as G
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    from native_same_s_falsify import native_inputs
    from native_time_audit_v2 import AuditV2
    G._install_cuda_property_compatibility()
    audit = None
    if a.model_contract:
        from curobo_certify import k4_model_hook as MH
        audit = MH.install_timeopt_model_hook(MH.planning_contract(json.loads(a.model_contract.read_text()), a.payload))
    elif a.payload:
        from curobo_certify import k4_payload as KP
        audit = KP.install_timeopt_payload_hook(a.payload)
    dev = torch.device("cuda:0")
    cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model=None, graph_planner_algorithm="rrt",
                                  enable_trajopt=True, enable_timeopt=True, enable_inverse_dynamics=True,
                                  use_cuda_graph=True, num_ik_seeds=4, num_trajopt_seeds=1, random_seed=1)
    rows = []
    with MotionPlanner(cfg) as planner:
        s0 = planner.default_joint_state.position.reshape(1, -1).clone()
        g0 = s0.clone(); g0[0, 0] += 0.1
        planner.plan_cspace(JointState.from_position(g0, joint_names=planner.joint_names),
                            JointState.from_position(s0, joint_names=planner.joint_names),
                            max_attempts=2, enable_graph_attempt=0, finetune_attempts=0)
        opt = planner.trajopt_solver._timeopt.optimizer
        dyn = opt._dynamics
        if (a.payload or a.model_contract) and not audit:
            raise SystemExit("HOOK_NOT_APPLIED")
        plim = [t.detach().cpu().numpy().astype(np.float64) for t in opt._limits[3:5]]
        eff = opt.config.robot.kinematics_config.joint_limits.effort.detach().cpu().numpy().astype(np.float64)
        A = AuditV2(torch, dev)
        dyn.setup_batch_size(batch_size=32768)
        for path in a.npz:
            raw = path.read_bytes()
            with np.load(path) as z:
                arr = {k: z[k].copy() for k in z.files}
            inp = native_inputs(arr, torch, dev)
            row = {"path": path.stem, "npz_sha256": hashlib.sha256(raw).hexdigest(), "grids": {}}
            for name, times in (("native_grid_512", arr["native_timing_time_grid"]),
                                ("S8193", arr["diagnostic_8193_times"]), ("S32768", arr["diagnostic_32768_times"])):
                ts = torch.as_tensor(np.asarray(times, dtype=np.float32), device=dev)
                out, _ = A.production(inp, ts)
                js = JointState(position=out[0, 0].contiguous(), velocity=out[1, 0].contiguous(),
                                acceleration=out[2, 0].contiguous())
                tau = dyn.compute_inverse_dynamics(js).detach().cpu().numpy().astype(np.float64)
                torch.cuda.synchronize()
                phys = np.maximum(tau / eff[1], tau / eff[0])
                plan = np.maximum(tau / plim[1], tau / plim[0])
                i = np.unravel_index(np.argmax(phys), phys.shape)
                row["grids"][name] = {"n": int(len(times)), "max_util_physical": float(phys.max()),
                                      "max_util_planning": float(plan.max()), "argmax_time": float(times[i[0]]),
                                      "argmax_joint": int(i[1]), "per_joint_physical": phys.max(0).tolist(),
                                      "exceed_physical_samples": int((phys > 1).any(1).sum()),
                                      "exceed_planning_samples": int((plan > 1).any(1).sum())}
            rows.append(row)
            print(json.dumps({"path": path.stem, **{k: round(v["max_util_physical"], 4) for k, v in row["grids"].items()}}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "NATIVE_SAMPLED_TORQUE_DIAGNOSTIC_NOT_PROOF", "gpu": torch.cuda.get_device_name(0), "payload": a.payload,
                                 "model_contract": str(a.model_contract) if a.model_contract else None,
                                 "hook_audit": ({k: v for k, v in audit[-1].items() if k != "links"} if audit else None),
                                 "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
