#!/usr/bin/env python3
"""Cost of a native GPU sampled torque check (baseline): cuRobo TimeOpt sample(times) + inverse dynamics at N points.
RTX 3090, planner built as the generator does (corrected model + 3 kg via k4_model_hook), one warm-up plan, then per N:
W warm-ups and R timed repeats of [sample N uniform-time states -> compute_inverse_dynamics -> max utilization -> sync].
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=int, nargs="+", default=[16, 64, 512, 4096, 32768])
    ap.add_argument("--warmups", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    import generate_native_pilot as G
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    from curobo_certify import k4_model_hook as MH
    G._install_cuda_property_compatibility()
    tc = MH.planning_contract(json.loads((ROOT / "docs/reports/k4/identified/panda_contract_identified_plus_hand.json").read_text()),
                              "PAYLOAD_SYNTH_3KG_CYL")
    MH.install_timeopt_model_hook(tc)
    cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model=None, graph_planner_algorithm="rrt", enable_trajopt=True,
                                  enable_timeopt=True, enable_inverse_dynamics=True, use_cuda_graph=True, num_ik_seeds=4,
                                  num_trajopt_seeds=1, random_seed=1)
    out = {"gpu": torch.cuda.get_device_name(0), "rows": []}
    with MotionPlanner(cfg) as p:
        s0 = p.default_joint_state.position.reshape(1, -1).clone(); g0 = s0.clone(); g0[0, 0] += 0.8; g0[0, 1] += 0.4
        p.plan_cspace(JointState.from_position(g0, joint_names=p.joint_names), JointState.from_position(s0, joint_names=p.joint_names),
                      max_attempts=2, enable_graph_attempt=0, finetune_attempts=0)
        to = p.trajopt_solver._timeopt.optimizer
        dyn = to._dynamics
        lo, hi = to._limits[3], to._limits[4]
        T = float(to._times[0, -1].item())
        for n in a.ns:
            to.reset_cuda_graph()
            to._sample_count = None
            times = torch.linspace(0, T, n, device=lo.device)[None].contiguous()
            dyn.setup_batch_size(batch_size=n)

            def check():
                st = to.sample(times)
                js = JointState(position=st.position[0].contiguous(), velocity=st.velocity[0].contiguous(),
                                acceleration=st.acceleration[0].contiguous())
                tau = dyn.compute_inverse_dynamics(js)
                u = torch.maximum(tau / hi, tau / lo).max()
                return float(u.item())          # includes the device->host sync a checker needs
            for _ in range(a.warmups):
                check()
            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter(); u = check(); ts.append(1e3 * (time.perf_counter() - t0))
            ts = np.array(ts)
            out["rows"].append({"n": n, "p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "max_util": u})
            print(json.dumps(out["rows"][-1]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
