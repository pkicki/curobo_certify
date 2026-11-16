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


def _cond(record: dict) -> str:
    return record.get("solver_condition") or (record.get("source_query") or {}).get("solver_condition")


# --- observation-only record of the TimeOpt configuration actually constructed -----
_CONSTRUCTED: list = []


def install_config_observer():
    """Wrap the TrajectoryTimeOpt symbol used by solver_timeopt so each constructed
    optimizer's config is recorded. The wrapper does not alter arguments."""
    import curobo._src.solver.solver_timeopt as st
    original = getattr(st, "_curobo_certify_original_timeopt", st.TrajectoryTimeOpt)
    st._curobo_certify_original_timeopt = original

    def observed(config, *args, **kwargs):
        _CONSTRUCTED.append(config)
        return original(config, *args, **kwargs)
    st.TrajectoryTimeOpt = observed


def config_audit(expected_scale: float) -> dict:
    """Compare the last constructed TimeOpt config with the pinned default constructor
    applied to the same arguments: only limit_scale/jerk_limit_scale may differ."""
    import dataclasses as dc
    from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
    if not _CONSTRUCTED:
        return {"status": "NOT_OBSERVED"}
    cfg = _CONSTRUCTED[-1]
    fields = {f.name for f in dc.fields(cfg)}
    kwargs = {f.name: getattr(cfg, f.name) for f in dc.fields(cfg) if f.name not in ("limit_scale", "jerk_limit_scale")}
    pinned = TrajectoryTimeOptCfg(**{k: v for k, v in kwargs.items() if k in fields})
    default_fields_differ = [f.name for f in dc.fields(cfg) if f.name not in ("limit_scale", "jerk_limit_scale")
                             and getattr(cfg, f.name) is not getattr(pinned, f.name) and getattr(cfg, f.name) != getattr(pinned, f.name)]
    ok = (abs(cfg.limit_scale - expected_scale) == 0 and abs(cfg.jerk_limit_scale - expected_scale) == 0
          and not default_fields_differ)
    return {"status": "PASS" if ok else "FAIL", "constructed_limit_scale": cfg.limit_scale,
            "constructed_jerk_limit_scale": cfg.jerk_limit_scale, "pinned_default_limit_scale": pinned.limit_scale,
            "pinned_default_jerk_limit_scale": pinned.jerk_limit_scale, "other_fields_differ": default_fields_differ,
            "constructions_observed": len(_CONSTRUCTED)}


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
    # Generic registered margin sweep conditions are strictly parsed and
    # re-constructed from Decimal; only the two TimeOpt planning scales vary.
    # Preserve the pinned 0.95 path as the unpatched baseline.
    from curobo_certify.margin_campaign import parse_label
    level = parse_label(condition)
    if level == 0.95:
        st.TrajectoryTimeOptCfg = TrajectoryTimeOptCfg
        return {"limit_scale": 0.95, "jerk_limit_scale": 0.95, "override": False}
    scale = float(level)
    def patched(**kw):
        return dataclasses.replace(TrajectoryTimeOptCfg(**kw), limit_scale=scale,
                                   jerk_limit_scale=scale)
    st.TrajectoryTimeOptCfg = patched
    return {"limit_scale": scale, "jerk_limit_scale": scale,
            "override": "runtime solver_timeopt.TrajectoryTimeOptCfg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--order-seed", type=int, default=None,
                    help="execute rows in a seeded random permutation (logged in summary.json)")
    a = ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    gpu = torch.cuda.get_device_name(0)
    compat = G._install_cuda_property_compatibility()
    install_config_observer()
    rows = json.loads(a.queries.read_text())
    if a.order_seed is not None:
        import random
        random.Random(a.order_seed).shuffle(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        path = a.out_dir / f"{row['source_id']}.json"
        if a.skip_existing and path.exists():
            records.append(json.loads(path.read_text()))
            continue
        from curobo_certify.query_pipeline import verify_query
        if not verify_query(row):
            raise SystemExit("SOURCE_QUERY_HASH_MISMATCH " + str(row.get("source_id")))
        cond = set_condition(row["solver_condition"])
        _CONSTRUCTED.clear()
        t0 = time.perf_counter()
        try:
            rec = G._run_row(row, a.out_dir, {"panda": "franka.yml"})
            audit = config_audit(cond["limit_scale"])
            if audit["status"] != "PASS":
                raise RuntimeError(f"TIMEOPT_CONFIG_AUDIT_FAILED {audit}")
            rec.update({"cuda_compatibility": compat, "solver_condition": row["solver_condition"],
                        "timeopt_scales": cond, "timeopt_config_audit": audit, "gpu": gpu,
                        "request_wall_seconds_incl_planner_setup": time.perf_counter() - t0,
                        "evidence_kind": "real_native_planner_timeopt_hardcase"})
            path.write_text(json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n")
            print(f"{row['source_id']}: {rec['status']} T={rec['native_duration_seconds']:.6g}s wall={time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as exc:  # retained failure
            G._dump_failure(path, row, "planner_or_timeopt", f"{type(exc).__name__}: {exc}",
                            cuda_compatibility=compat, solver_condition=row["solver_condition"], gpu=gpu,
                            timeopt_scales=cond, timeopt_config_audit=config_audit(cond["limit_scale"]),
                            request_wall_seconds_incl_planner_setup=time.perf_counter() - t0)
            print(f"{row['source_id']}: FAILED_RETAINED {type(exc).__name__}: {exc}", flush=True)
        records.append(json.loads(path.read_text()))
    summary = {"query_manifest_sha256": hashlib.sha256(a.queries.read_bytes()).hexdigest(), "gpu": gpu,
               "attempted": len(rows),
               "native_success": sum(r.get("planner_success") is True for r in records),
               "failed_retained": sum(r.get("status") == "FAILED_RETAINED" for r in records),
               "by_condition": {c: {"attempted": sum(_cond(r) == c for r in records),
                                    "native_success": sum(_cond(r) == c and r.get("planner_success") is True for r in records),
                                    "failed_retained": sum(_cond(r) == c and r.get("status") == "FAILED_RETAINED" for r in records)}
                                for c in sorted({row["solver_condition"] for row in rows})},
               "order": [row["source_id"] for row in rows], "order_seed": a.order_seed}
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
