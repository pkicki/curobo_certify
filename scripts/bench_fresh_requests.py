#!/usr/bin/env python3
"""Paired fresh-request latency: native planner+TimeOpt with and without certification.

For each frozen query (one warm MotionPlanner instance per query, same settings as
the generator): 50 warmup requests, then N repeats, each a seeded random AB or BA
pair of
  A  plan_cspace -> CUDA sync -> CPU-visible native success decision
  B  plan_cspace -> CUDA sync -> device->host copy of effective controls/timing ->
     compiled native FP32 bound -> compiled K3 producer (adaptive) -> independent
     verifier -> CPU-visible certificate decision
Wall time by perf_counter_ns around the whole request. Output byte hashes are
recorded for every request (determinism check; differences are reported, not hidden).
Scope: FRESH_NATIVE_TOTAL and ADDED latency, EXPLORATORY population (not Dev40).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from curobo_certify import compiled_k3 as K  # noqa: E402
import generate_hardcase_population as GH  # noqa: E402
import generate_native_pilot as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--warmups", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--contract", choices=("physical", "native"), default="native")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--threads", nargs="+", type=int, default=[1],
                    help="certification arms: one arm per OpenMP thread count (paired with native-only arm A)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    compat = G._install_cuda_property_compatibility()
    gpu = torch.cuda.get_device_name(0)
    rows = json.loads(a.queries.read_text())
    if a.ids:
        rows = [r for r in rows if r["source_id"] in a.ids]
    rows = rows[: a.limit]
    tmp = tempfile.TemporaryDirectory()
    builds = [K.build("k3_certifier.cpp", Path(tmp.name) / "c.so"), K.build("k3_verifier.cpp", Path(tmp.name) / "v.so")]
    P, V = K.Producer(Path(tmp.name) / "c.so"), K.Verifier(Path(tmp.name) / "v.so")
    rng = np.random.default_rng(a.seed)
    results = []
    for row in rows:
        cond = GH.set_condition(row.get("solver_condition", "default_margin_0p95"))
        scene = None if row["scene_id"] == "empty" else row["scene_id"] + ".yml"
        cfg = MotionPlannerCfg.create(robot="franka.yml", scene_model=scene, graph_planner_algorithm="rrt",
                                      enable_trajopt=True, enable_timeopt=True, enable_inverse_dynamics=True,
                                      use_cuda_graph=True, num_ik_seeds=4, num_trajopt_seeds=1,
                                      random_seed=int(row["seed"] + row["query_index"]))
        record = {"source_id": row["source_id"], "solver_condition": row.get("solver_condition"), "timeopt_scales": cond}
        with MotionPlanner(cfg) as planner:
            dev = planner.default_joint_state.position.device
            start = torch.tensor(row["start_joint_positions_rad"], device=dev, dtype=torch.float32)[None]
            goal = torch.tensor(row["goal_joint_positions_rad"], device=dev, dtype=torch.float32)[None]
            cur = JointState.from_position(start, joint_names=planner.joint_names)
            tgt = JointState.from_position(goal, joint_names=planner.joint_names)
            lim_t = planner.kinematics.config.get_joint_limits()
            limits = {f: np.minimum(np.abs(getattr(lim_t, f)[0].cpu().numpy()), np.abs(getattr(lim_t, f)[1].cpu().numpy())).astype(np.float64)
                      for f in ("velocity", "acceleration", "jerk")}

            def request(certify: bool, threads: int = 1):
                t0 = time.perf_counter_ns()
                res = planner.plan_cspace(tgt, cur, max_attempts=2, enable_graph_attempt=0, finetune_attempts=0)
                torch.cuda.synchronize(dev)
                ok = bool(res is not None and res.success.any().item())
                status, digest, stages = "NATIVE_FAILURE", None, {}
                if ok:
                    spline = res.spline_trajectory
                    t1 = time.perf_counter_ns()
                    g = spline.control_points[0].detach().cpu().numpy().copy()
                    tc = spline.timing.controls[0].detach().cpu().numpy().copy()
                    t2 = time.perf_counter_ns()
                    digest = hashlib.sha256(g.tobytes() + tc.tobytes()).hexdigest()
                    status = "NATIVE_SUCCESS"
                    if certify:
                        tighten = K.native_bounds(P, g, tc, threads=threads) if a.contract == "native" else None
                        t3 = time.perf_counter_ns()
                        args = (g, tc, start[0].cpu().numpy(), goal[0].cpu().numpy(), limits)
                        r = P.certify(*args, tighten=tighten, max_depth=12, max_nodes=8192, threads=threads)
                        t4 = time.perf_counter_ns()
                        if r["status"] == "CERTIFIED":
                            v = V.verify(*args, r["leaves"], K.input_digest(*args, tighten), tighten=tighten, threads=threads)
                            status = "CERTIFIED" if v["pass"] else "REJECTED_BY_VERIFIER"
                        else:
                            status = r["status"]
                        t5 = time.perf_counter_ns()
                        stages = {"plan_ms": (t1 - t0) / 1e6, "d2h_ms": (t2 - t1) / 1e6, "native_bound_ms": (t3 - t2) / 1e6,
                                  "producer_ms": (t4 - t3) / 1e6, "verifier_ms": (t5 - t4) / 1e6}
                return (time.perf_counter_ns() - t0) / 1e6, status, digest, stages

            arms = ["A"] + [f"B{th}" for th in a.threads]
            for _ in range(a.warmups):
                for arm in arms:
                    request(arm != "A", int(arm[1:]) if arm != "A" else 1)
            reps = []
            for r in range(a.repeats):
                order = list(arms)
                rng.shuffle(order)
                pair = {}
                for arm in order:
                    ms, st, dg, stg = request(arm != "A", int(arm[1:]) if arm != "A" else 1)
                    pair[arm] = {"ms": ms, "status": st, "digest": dg, "stages": stg}
                pair["order"] = ",".join(order)
                cert = {pair[x]["status"] for x in arms if x != "A"}
                if len(cert) > 1 and not any(pair[x]["status"] == "NATIVE_FAILURE" for x in arms):
                    raise SystemExit(f"certificate status differs across thread arms: {pair}")
                pair["B"] = pair[arms[1]]  # first certification arm keeps the legacy key
                reps.append(pair)
        digests = {p[arm]["digest"] for p in reps for arm in arms}
        record.update({"repeats": reps, "distinct_output_digests": len(digests),
                       "added_ms": [p["B"]["ms"] - p["A"]["ms"] for p in reps],
                       "added_ms_by_arm": {arm: [p[arm]["ms"] - p["A"]["ms"] for p in reps] for arm in arms[1:]},
                       "stage_by_arm": {arm: [p[arm]["stages"] for p in reps] for arm in arms[1:]},
                       "native_ms": [p["A"]["ms"] for p in reps], "total_ms": [p["B"]["ms"] for p in reps]})
        print(json.dumps({"id": row["source_id"], "native_p50": float(np.median(record["native_ms"])),
                          "added_p50": float(np.median(record["added_ms"])), "status": reps[0]["B"]["status"],
                          "digests": len(digests)}), flush=True)
        results.append(record)
    out = {"scope": "FRESH_REQUEST_PAIRED_LATENCY_EXPLORATORY", "gpu": gpu, "contract": a.contract,
           "threads_arms": a.threads,
           "warmups": a.warmups, "repeats": a.repeats, "builds": builds, "cuda_compatibility": compat,
           "query_manifest_sha256": hashlib.sha256(a.queries.read_bytes()).hexdigest(), "rows": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
