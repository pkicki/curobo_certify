"""Deterministic, scope-aware planning-margin experiment design.

The data here are source queries, not solutions or certified trajectories.
Only the frozen GPU worker is allowed to instantiate cuRobo/TimeOpt.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .query_pipeline import _canonical_hash, verify_query

UPSTREAM_SHA = "c301cb00a87012c2808b1a60649c4065c9b3c95a"
DEFAULT_SCALES = ("0.90", "0.95", "0.975", "0.99", "1.00")
HOME = (0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8)
LIMITS = ((-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
          (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
          (-2.8973, 2.8973))
SCENES = ("empty", "collision_table", "collision_test", "collision_primitives_3d")
MAGNITUDES = {"small": (0.06, 0.22), "medium": (0.3, 0.7), "large": (0.8, 1.4)}


def as_margin(value: str | Decimal) -> Decimal:
    """Canonical Decimal on (0,1]; reject exponent and ambiguous formatting."""
    s = str(value).strip()
    if not s or "e" in s.lower() or len(s) > 20:
        raise ValueError("margin must be an ordinary finite decimal on (0,1]")
    try:
        d = Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"invalid margin {s!r}") from exc
    if not d.is_finite() or not Decimal(0) < d <= Decimal(1):
        raise ValueError("margin must be in (0,1]")
    return d


def margin_label(value: str | Decimal) -> str:
    d = as_margin(value)
    s = format(d.quantize(Decimal("0.001")), "f")
    if Decimal(s) != d:
        raise ValueError("margin needs <=3 decimal places")
    return "margin_" + s.replace(".", "p")


def parse_label(label: str) -> Decimal:
    if label == "default_margin_0p95":
        return Decimal("0.95")
    if label == "margin_arm_1p00":
        return Decimal("1.00")
    if not isinstance(label, str) or not label.startswith("margin_"):
        raise ValueError("unsupported planning margin condition")
    s = label[len("margin_"):].replace("p", ".")
    d = as_margin(s)
    if margin_label(d) != label:
        raise ValueError("noncanonical margin label")
    return d


def scales(values: Iterable[str]) -> tuple[Decimal, ...]:
    result = tuple(as_margin(s) for s in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("empty or duplicated margin scales")
    for d in result:
        parse_label(margin_label(d))
    return tuple(sorted(result))


def freeze(seed: int, per_cell: int, margins: Iterable[str] = DEFAULT_SCALES) -> list[dict]:
    if per_cell < 1 or per_cell > 100 or not isinstance(seed, int):
        raise ValueError("invalid seed or per-cell count")
    levels = scales(margins)
    rng = random.Random(seed)
    result: list[dict] = []
    index = 0
    for scene in SCENES:
        for magnitude, (lo, hi) in MAGNITUDES.items():
            for _ in range(per_cell):
                goal = []
                for j, home in enumerate(HOME):
                    lower, upper = LIMITS[j]
                    center, radius = (lower + upper) / 2, 0.45 * (upper - lower)
                    position = home + rng.choice((-1, 1)) * rng.uniform(lo, hi)
                    goal.append(min(max(position, center - radius), center + radius))
                for margin in levels:
                    label = margin_label(margin)
                    row = {
                        "schema": "timeopt_native_source_query_v1",
                        "robot": "panda",
                        "scene_id": scene,
                        "scene_stratum": scene,
                        "magnitude_stratum": magnitude,
                        "seed": seed,
                        "query_index": index,
                        "pair_id": f"margin-{seed}-{index:04d}",
                        "solver_condition": label,
                        "planning_margin": str(margin),
                        "source_id": f"margin-{seed}-{index:04d}-{label}",
                        "goal_delta_rad": [g-h for g, h in zip(goal, HOME)],
                        "start_joint_positions_rad": list(HOME),
                        "goal_joint_positions_rad": goal,
                        "query_resolution": "FROZEN_EXPLICIT_CSPACE",
                        "planner": "pinned_curobo_motion_planner.plan_cspace",
                        "timeopt": "pinned_curobo.TrajectoryTimeOpt",
                        "source_geometry_contract": (
                            "selected planner-integrated TimeOpt output: cloned effective cubic geometry, "
                            "unchanged 32 timing controls, physical actuator limits"),
                    }
                    row["source_query_sha256"] = _canonical_hash(row)
                    if not verify_query(row):
                        raise AssertionError("failed to freeze query")
                    result.append(row)
                index += 1
    return result


def check_manifest(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty margin manifest")
    unique = set()
    by_pair: dict[str, list[dict]] = {}
    all_levels = set()
    for row in rows:
        if not isinstance(row, dict) or not verify_query(row):
            raise ValueError("bad source query hash")
        if row.get("robot") != "panda" or row.get("schema") != "timeopt_native_source_query_v1":
            raise ValueError("unsupported query model")
        if row["source_id"] in unique:
            raise ValueError("duplicate source id")
        unique.add(row["source_id"])
        d = parse_label(row["solver_condition"])
        if d != as_margin(row["planning_margin"]):
            raise ValueError("condition and margin disagree")
        all_levels.add(d)
        by_pair.setdefault(row["pair_id"], []).append(row)
    for pair_id, group in by_pair.items():
        if {as_margin(g["planning_margin"]) for g in group} != all_levels:
            raise ValueError(f"incomplete arms for pair {pair_id}")
        keep = ("robot", "scene_id", "scene_stratum", "magnitude_stratum", "seed",
                "query_index", "goal_delta_rad", "start_joint_positions_rad",
                "goal_joint_positions_rad")
        anchor = {k: group[0][k] for k in keep}
        if any({k: g[k] for k in keep} != anchor for g in group[1:]):
            raise ValueError(f"geometrically non-matched pair {pair_id}")
    return {"scope": "FROZEN_MARGIN_SOURCE_QUERIES_NOT_TRAJECTORIES",
            "query_count": len(rows), "pairs": len(by_pair),
            "margins": [str(x) for x in sorted(all_levels)],
            "upstream_sha": UPSTREAM_SHA}


def dump_manifest(rows: list[dict]) -> bytes:
    check_manifest(rows)
    return (json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
