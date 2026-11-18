#!/usr/bin/env python3
"""K4 HARD CASES (1b): cuRobo's NATIVE float32 sampled torque check on the near-limit ladder cases (GPU, RTX 3090).

For each case of k4_hard_nearlimit (base NPZ + scaled float32 timing controls), the base NPZ arrays are reused with
timing_controls := the case bytes and every time array divided by the time scale lambda (x -> lambda^2 x exactly up to
float32 rounding of the controls, so t -> t / lambda). cuRobo's production sampler (TimeOpt sample kernel) evaluates
states at TimeOpt's output grid, S8193 and S32768 uniform-time samples; TimeOpt's own GPU inverse dynamics (planner built
with the corrected model + payload hook, as in planning) gives torque. Verdict: max utilization <= 1 (planning limits).
SAMPLED + NUMERICAL evidence only; compared against the rigorous bracket of the same bytes.
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
    ap.add_argument("--cases", type=Path, nargs="+", required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--model-contract", type=Path, required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    import generate_native_pilot as G
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    from curobo_certify import k4_model_hook as MH
    from native_same_s_falsify import native_inputs
    from native_time_audit_v2 import AuditV2
    G._install_cuda_property_compatibility()
    tc_ = MH.planning_contract(json.loads(a.model_contract.read_text()), a.payload)
    audit = MH.install_timeopt_model_hook(tc_)
    dev = torch.device("cuda:0")
    cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model=None, graph_planner_algorithm="rrt", enable_trajopt=True,
                                  enable_timeopt=True, enable_inverse_dynamics=True, use_cuda_graph=True, num_ik_seeds=4,
                                  num_trajopt_seeds=1, random_seed=1)
    rows = []
    with MotionPlanner(cfg) as planner:
        s0 = planner.default_joint_state.position.reshape(1, -1).clone(); g0 = s0.clone(); g0[0, 0] += 0.1
        planner.plan_cspace(JointState.from_position(g0, joint_names=planner.joint_names),
                            JointState.from_position(s0, joint_names=planner.joint_names),
                            max_attempts=2, enable_graph_attempt=0, finetune_attempts=0)
        if not audit or not audit[-1]["masses_match"]:
            raise SystemExit("HOOK_NOT_APPLIED")
        opt = planner.trajopt_solver._timeopt.optimizer
        dyn = opt._dynamics
        plo, phi = [t.detach().cpu().numpy().astype(np.float64) for t in opt._limits[3:5]]
        A = AuditV2(torch, dev)
        dyn.setup_batch_size(batch_size=32768)
        for cf in a.cases:
            for case in json.loads(cf.read_text())["rows"]:
                with np.load(a.npz_dir / f"{case['base']}.npz") as z:
                    arr = {k: z[k].copy() for k in z.files}
                lam = 1.0 / case["time_scale"]
                arr["timing_controls"] = np.array([float.fromhex(h) for h in case["timing_controls_hex"]], dtype=np.float32)
                for k in ("native_timing_time_grid", "diagnostic_8193_times", "diagnostic_32768_times"):
                    arr[k] = (arr[k].astype(np.float64) / lam).astype(np.float32)
                inp = native_inputs(arr, torch, dev)
                res = {}
                for name, key in (("timeopt_grid", "native_timing_time_grid"), ("S8193", "diagnostic_8193_times"),
                                  ("S32768", "diagnostic_32768_times")):
                    ts = torch.as_tensor(arr[key], device=dev)
                    out, _ = A.production(inp, ts)
                    js = JointState(position=out[0, 0].contiguous(), velocity=out[1, 0].contiguous(), acceleration=out[2, 0].contiguous())
                    tau = dyn.compute_inverse_dynamics(js).detach().cpu().numpy().astype(np.float64)
                    res[name] = float(np.maximum(tau / phi, tau / plo).max())
                row = {"base": case["base"], "eps_target": case["eps_target"], "tc_sha256": case["tc_sha256"], "truth": case["truth"],
                       "lb": case["lb"], "ub": case["ub"], "native": res}
                rows.append(row)
                print(json.dumps({"base": case["base"][-18:], "eps": case["eps_target"], "truth": case["truth"],
                                  **{k: v - 1 for k, v in res.items()}}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "K4_HARD_NEARLIMIT_NATIVE_FLOAT32", "planning_limits_float32": [plo.tolist(), phi.tolist()],
                                 "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
