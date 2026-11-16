#!/usr/bin/env python3
"""Freeze a seeded Panda hard-case exploratory query population BEFORE solving.

Strata: 4 scenes x 3 motion magnitudes (small/medium/large joint offsets from
the pinned planner default state) x 3 queries. Each query is emitted for two
separately labelled solver conditions:
  * default_margin_0p95 - unchanged pinned TimeOpt configuration;
  * margin_arm_1p00     - the campaign-registered exploratory margin arm
                          (limit_scale = jerk_limit_scale = 1.00), applied at
                          runtime only; its outputs are never pooled with default.
This is an EXPLORATORY population (not Dev40/Holdout100/Confirm300).
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.query_pipeline import _canonical_hash  # noqa: E402

PANDA_HOME = [0.0, -1.3, 0.0, -2.5, 0.0, 1.5, 0.8]
# franka.yml position limits (pinned cuRobo), shrunk by 10% towards the centre.
PANDA_LIMITS = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
                (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
SCENES = ("empty", "collision_table", "collision_test", "collision_primitives_3d")
MAGNITUDES = {"small": (0.06, 0.22), "medium": (0.3, 0.7), "large": (0.8, 1.4)}
CONDITIONS = ("default_margin_0p95", "margin_arm_1p00")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    index = 0
    for scene in SCENES:
        for mag, (lo, hi) in MAGNITUDES.items():
            for rep in range(a.per_cell):
                goal = []
                for j, h in enumerate(PANDA_HOME):
                    l, u = PANDA_LIMITS[j]
                    c, r = 0.5 * (l + u), 0.45 * (u - l)
                    g = h + rng.choice((-1, 1)) * rng.uniform(lo, hi)
                    goal.append(min(max(g, c - r), c + r))
                for cond in CONDITIONS:
                    row = {"schema": "timeopt_native_source_query_v1", "robot": "panda", "scene_id": scene,
                           "scene_stratum": scene, "magnitude_stratum": mag, "seed": a.seed, "query_index": index,
                           "pair_id": f"hard-{a.seed}-{index:03d}", "solver_condition": cond,
                           "source_id": f"hard-{a.seed}-{index:03d}-{cond}",
                           "goal_delta_rad": [g - h for g, h in zip(goal, PANDA_HOME)],
                           "start_joint_positions_rad": PANDA_HOME, "goal_joint_positions_rad": goal,
                           "query_resolution": "FROZEN_EXPLICIT_CSPACE",
                           "planner": "pinned_curobo_motion_planner.plan_cspace",
                           "timeopt": "pinned_curobo.TrajectoryTimeOpt",
                           "source_geometry_contract": "selected planner-integrated TimeOpt result exports its complete effective cubic spline and unchanged timing controls"}
                    row["source_query_sha256"] = _canonical_hash(row)
                    rows.append(row)
                index += 1
    data = (json.dumps(rows, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    side = {"schema": "timeopt_panda_hardcase_queries_v1", "scope": "EXPLORATORY_HARD_CASE_POPULATION_NOT_FORMAL_COHORT",
            "seed": a.seed, "query_file_sha256": hashlib.sha256(data).hexdigest(), "queries": index,
            "rows": len(rows), "conditions": CONDITIONS, "scenes": SCENES, "magnitudes": MAGNITUDES,
            "upstream_sha": "c301cb00a87012c2808b1a60649c4065c9b3c95a",
            "payload": "NOT_VARIED: pinned franka.yml has no verified alternative payload configuration in this pass"}
    Path(str(a.out) + ".manifest.json").write_text(json.dumps(side, sort_keys=True, indent=2) + "\n")
    print(json.dumps(side))


if __name__ == "__main__":
    main()
