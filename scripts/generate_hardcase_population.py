#!/usr/bin/env python3
"""Solve a frozen hard-case query manifest with the pinned native planner+TimeOpt.

Reuses ``generate_native_pilot._run_row`` unchanged (same planner settings,
exports, parity/diagnostic sampling, hashing). For rows whose solver_condition
is ``margin_arm_1p00`` the TimeOpt planning scales are set to 1.00 by a
runtime-only override of the config class referenced inside
``curobo._src.solver.solver_timeopt`` (no source file is modified); the record
stores the condition. Every failure is retained in the denominator.
"""
import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import generate_native_pilot as G  # noqa: E402


def set_condition(condition: str):
    import curobo._src.solver.solver_timeopt as st
    from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
    if condition == "default_margin_0p95":
        st.TrajectoryTimeOptCfg = TrajectoryTimeOptCfg
        return {"limit_scale": 0.95, "jerk_limit_scale": 0.95, "override": False}
    if condition == "margin_arm_1p00":
        def patched(**kw):
            return dataclasses.replace(TrajectoryTimeOptCfg(**kw), limit_scale=1.0, jerk_limit_scale=1.0)
        st.TrajectoryTimeOptCfg = patched
        return {"limit_scale": 1.0, "jerk_limit_scale": 1.0, "override": "runtime solver_timeopt.TrajectoryTimeOptCfg"}
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    compat = G._install_cuda_property_compatibility()
    rows = json.loads(a.queries.read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        path = a.out_dir / f"{row['source_id']}.json"
        if a.skip_existing and path.exists():
            records.append(json.loads(path.read_text()))
            continue
        cond = set_condition(row["solver_condition"])
        t0 = time.perf_counter()
        try:
            rec = G._run_row(row, a.out_dir, {"panda": "franka.yml"})
            rec.update({"cuda_compatibility": compat, "solver_condition": row["solver_condition"],
                        "timeopt_scales": cond, "gpu": gpu, "evidence_kind": "real_native_planner_timeopt_hardcase"})
            path.write_text(json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n")
            print(f"{row['source_id']}: {rec['status']} T={rec['native_duration_seconds']:.6g}s wall={time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as exc:  # retained failure
            G._dump_failure(path, row, "planner_or_timeopt", f"{type(exc).__name__}: {exc}",
                            cuda_compatibility=compat, solver_condition=row["solver_condition"], gpu=gpu)
            print(f"{row['source_id']}: FAILED_RETAINED {type(exc).__name__}: {exc}", flush=True)
        records.append(json.loads(path.read_text()))
    summary = {"query_manifest_sha256": hashlib.sha256(a.queries.read_bytes()).hexdigest(), "gpu": gpu,
               "attempted": len(rows),
               "native_success": sum(r.get("planner_success") is True for r in records),
               "failed_retained": sum(r.get("status") == "FAILED_RETAINED" for r in records),
               "by_condition": {c: {"attempted": sum(r["source_query"]["solver_condition"] == c if "source_query" in r else r.get("solver_condition") == c for r in records)}
                                for c in ("default_margin_0p95", "margin_arm_1p00")}}
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
