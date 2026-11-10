"""Deterministic source-query descriptions for native TimeOpt pilot generation.

This module creates *queries*, never solutions. Query files can therefore be
hashed and frozen before planner execution. It deliberately does not infer a
robot model from a similarly named cuRobo config (UR10e is not UR5e).
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

PILOT_SEED = 20261110


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pilot_queries(seed: int = PILOT_SEED) -> list[dict[str, Any]]:
    """Make eight executable Panda and eight unresolved UR5e source requests.

    Query coordinates are offsets from each pinned robot config's declared
    default active state. The runner resolves these only after checking model
    identity and records an unresolved model as a retained planner-stage
    failure. The UR5e requests are intentionally marked unresolved until a
    verified model is provided. No trajectory coefficients or cached paths
    are reused.
    """
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    panda_home = [0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]
    for robot in ("panda", "ur5e"):
        for index in range(8):
            # Deterministic but source-distinct changes across active joints.
            dof = 7 if robot == "panda" else 6
            direction = [rng.choice((-1, 1)) * rng.uniform(0.06, 0.22) for _ in range(dof)]
            # Keep Panda fingers fixed; the seven arm joints carry the query.
            record = {
                "schema": "timeopt_native_source_query_v1",
                "source_id": f"pilot-{robot}-{index:02d}",
                "robot": robot,
                "scene_id": ("empty" if index < 3 else "collision_table" if index < 6 else "collision_test"),
                "scene_stratum": ("obstacle_free" if index < 3 else "moderately_cluttered" if index < 6 else "difficult"),
                "seed": seed,
                "query_index": index,
                "goal_delta_rad": direction,
                "start_joint_positions_rad": panda_home if robot == "panda" else None,
                "goal_joint_positions_rad": ([panda_home[i] + direction[i] for i in range(dof)]
                                              if robot == "panda" else None),
                "query_resolution": ("FROZEN_EXPLICIT_CSPACE" if robot == "panda"
                                     else "BLOCKED_NO_VERIFIED_MODEL_ASSET"),
                "planner": "pinned_curobo_motion_planner.plan_cspace",
                "timeopt": "pinned_curobo.TrajectoryTimeOpt",
                "source_geometry_contract": "selected planner-integrated TimeOpt result exports its complete effective cubic spline and unchanged timing controls",
            }
            record["source_query_sha256"] = _canonical_hash(record)
            records.append(record)
    return records


def verify_query(record: dict[str, Any]) -> bool:
    """Check a frozen query's canonical content hash."""
    expected = record.get("source_query_sha256")
    body = {key: value for key, value in record.items() if key != "source_query_sha256"}
    return isinstance(expected, str) and _canonical_hash(body) == expected
