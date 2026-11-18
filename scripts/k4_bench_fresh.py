#!/usr/bin/env python3
"""K4 P5 fresh-request paired latency (RTX 3090 planner + CPU certification).

Per frozen primary source query (one warm MotionPlanner per query, generator settings): W warmups, then
R repeats, each a seeded random permutation of arms
  A  native plan_cspace -> sync -> CPU-visible success
  B  A + device->host copy + compiled MODEL_K4 producer (k4_rnea.cpp: spans, Bernstein, tm1, witnesses)
       + independent compiled world-frame verifier (k4_verify.cpp)       [K4 only]
  C  B + MODEL_K3 compiled producer + verifier (frozen K3 code)           [K3 + K4]
All compiled stages use --threads OpenMP threads. Stage times are recorded. Native outputs differ run to run
(planner nondeterminism): every request's own output is certified and its digest recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--warmups", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    import generate_hardcase_population as GH
    import generate_native_pilot as G
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    from curobo_certify import compiled_k3 as K
    from curobo_certify import k4_compiled as KC
    from curobo_certify.k4_panda_model import PandaModel, load_contract
    from curobo_certify.k4_reuse import GeometryCache
    from curobo_certify.k4_verify import verify_k4
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    compat = G._install_cuda_property_compatibility()
    gpu = torch.cuda.get_device_name(0)
    c = load_contract(ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
    M = PandaModel(c)
    lo = np.array([float(Q(v)) for v in c["model"]["effort_lower"]])
    hi = np.array([float(Q(v)) for v in c["model"]["effort_upper"]])
    tmp = tempfile.TemporaryDirectory()
    kb = {"producer": KC.build(Path(tmp.name) / "k4.so"), "verifier": KC.build_verifier(Path(tmp.name) / "k4v.so")}
    KP = KC.CompiledPath(M, Path(tmp.name) / "k4.so")
    KVf = KC.CompiledVerifier(c, Path(tmp.name) / "k4v.so")
    builds = [K.build("k3_certifier.cpp", Path(tmp.name) / "c.so"), K.build("k3_verifier.cpp", Path(tmp.name) / "v.so")]
    P3, V3 = K.Producer(Path(tmp.name) / "c.so"), K.Verifier(Path(tmp.name) / "v.so")
    rows = [r for r in json.loads(a.queries.read_text()) if r["source_id"] in set(a.ids)]
    rng = np.random.default_rng(a.seed)
    results = []
    for row in rows:
        cond = GH.set_condition(row["solver_condition"])
        scene = None if row["scene_id"] == "empty" else row["scene_id"] + ".yml"
        cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model=scene, graph_planner_algorithm="rrt",
                                      enable_trajopt=True, enable_timeopt=True, enable_inverse_dynamics=True,
                                      use_cuda_graph=True, num_ik_seeds=4, num_trajopt_seeds=1,
                                      random_seed=int(row["seed"] + row["query_index"]))
        with MotionPlanner(cfg) as planner:
            dev = planner.default_joint_state.position.device
            start = torch.tensor(row["start_joint_positions_rad"], device=dev, dtype=torch.float32)[None]
            goal = torch.tensor(row["goal_joint_positions_rad"], device=dev, dtype=torch.float32)[None]
            cur = JointState.from_position(start, joint_names=planner.joint_names)
            tgt = JointState.from_position(goal, joint_names=planner.joint_names)
            lim_t = planner.kinematics.config.get_joint_limits()
            klim = {f: np.minimum(np.abs(getattr(lim_t, f)[0].cpu().numpy()), np.abs(getattr(lim_t, f)[1].cpu().numpy())).astype(np.float64)
                    for f in ("velocity", "acceleration", "jerk")}

            def request(arm):
                t0 = time.perf_counter_ns()
                res = planner.plan_cspace(tgt, cur, max_attempts=2, enable_graph_attempt=0, finetune_attempts=0)
                torch.cuda.synchronize(dev)
                ok = bool(res is not None and res.success.any().item())
                st, stages, digest = ("NATIVE_SUCCESS" if ok else "NATIVE_FAILURE"), {}, None
                if ok and arm != "A":
                    sp_ = res.spline_trajectory
                    t1 = time.perf_counter_ns()
                    g = sp_.control_points[0].detach().cpu().numpy().copy()
                    tc = sp_.timing.controls[0].detach().cpu().numpy().copy()
                    digest = hashlib.sha256(g.tobytes() + tc.tobytes()).hexdigest()
                    t2 = time.perf_counter_ns()
                    r4 = KP.certify(g, tc, lo, hi, threads=a.threads)
                    t4 = time.perf_counter_ns()
                    k4 = r4["status"]
                    if k4 == "CERTIFIED":
                        k4 = "CERTIFIED" if KVf.verify(g, tc, r4, lo, hi, threads=a.threads)["pass"] else "REJECTED_BY_VERIFIER"
                    t5 = time.perf_counter_ns()
                    stages = {"plan_ms": (t1 - t0) / 1e6, "d2h_ms": (t2 - t1) / 1e6,
                              "k4_producer_ms": (t4 - t2) / 1e6, "k4_verifier_ms": (t5 - t4) / 1e6}
                    st = "K4_" + k4
                    if arm == "C":
                        args = (g, tc, start[0].cpu().numpy(), goal[0].cpu().numpy(), klim)
                        r3 = P3.certify(*args, max_depth=12, max_nodes=8192, threads=a.threads)
                        k3 = r3["status"]
                        if k3 == "CERTIFIED":
                            k3 = "CERTIFIED" if V3.verify(*args, r3["leaves"], K.input_digest(*args, None), threads=a.threads)["pass"] else "REJECTED_BY_VERIFIER"
                        stages["k3_ms"] = (time.perf_counter_ns() - t5) / 1e6
                        st = f"K3_{k3}+K4_{k4}"
                return (time.perf_counter_ns() - t0) / 1e6, st, digest, stages

            for _ in range(a.warmups):
                for arm in ("A", "B", "C"):
                    request(arm)
            reps = []
            for _ in range(a.repeats):
                order = list(rng.permutation(["A", "B", "C"]))
                rep = {"order": ",".join(order)}
                for arm in order:
                    ms, st, dg, stg = request(arm)
                    rep[arm] = {"ms": ms, "status": st, "digest": dg, "stages": stg}
                reps.append(rep)
        rec = {"source_id": row["source_id"], "timeopt_scales": cond, "repeats": reps,
               "added_ms_B": [r["B"]["ms"] - r["A"]["ms"] for r in reps],
               "added_ms_C": [r["C"]["ms"] - r["A"]["ms"] for r in reps],
               "native_ms": [r["A"]["ms"] for r in reps],
               "distinct_digests": len({r[x]["digest"] for r in reps for x in ("B", "C") if r[x]["digest"]})}
        results.append(rec)
        print(json.dumps({"id": row["source_id"], "native_p50": float(np.median(rec["native_ms"])),
                          "added_B_p50": float(np.median(rec["added_ms_B"])), "added_C_p50": float(np.median(rec["added_ms_C"])),
                          "status": reps[0]["C"]["status"]}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "K4_FRESH_REQUEST_PAIRED_LATENCY_COMPILED", "gpu": gpu,
                                 "warmups": a.warmups, "repeats": a.repeats, "threads": a.threads, "k4_build": kb, "k3_builds": builds,
                                 "cuda_compatibility": compat, "rows": results}) + "\n")


if __name__ == "__main__":
    main()
