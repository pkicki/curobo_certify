"""K4 exploratory Panda torque population (K4-Panda-Dev40), frozen BEFORE any solve.

Separate from the K3 formal qualification (Panda-Dev40/Holdout100/Confirm300 untouched; their pools are
excluded by the leakage audit). Same generator distribution as the K3 studies (4 scenes x 3 motion
magnitudes, planner HOME start), new SHA-256-derived seed. Arms on identical start/goal/scene:
  primary   0.95  (unpatched cuRobo TimeOpt constructor)
  secondary 0.99  (registered paired arm)
  stress    1.00  (separately reported stress arm, never pooled)
Payload: PAYLOAD_DEFAULT_0P01KG only (cuRobo attached_object default); 3 kg / 5 kg are BLOCKED_PAYLOAD.
Selection: deterministic first-K (K = 40) primary successes over an ordered oversubscribed pool of 70;
secondary/stress arms solved for exactly the attempted source queries; nothing replaced.
Repeats: the primary arm of every attempted source query is solved 3 times (repeat 0 is the registered
frozen output; repeats 1-2 are nondeterminism diagnostics, certified separately and never substituted).
"""
from __future__ import annotations

import hashlib
import random

from .formal_cohorts import CELLS, _goal, dump, first_k_selection, leakage_audit  # noqa: F401
from .margin_campaign import HOME, margin_label
from .query_pipeline import _canonical_hash, verify_query

PROTOCOL = "k4_panda_exploratory_v1"
COHORT = "K4-Panda-Dev40"
K, POOL, REPEATS = 40, 70, 3
ARMS = (("primary", "0.95"), ("secondary", "0.99"), ("stress", "1.00"))
PAYLOAD = "PAYLOAD_DEFAULT_0P01KG"
SALT = "curobo_certify/k4_panda_exploratory_v1/"


def seed() -> int:
    return int.from_bytes(hashlib.sha256((SALT + COHORT).encode()).digest()[:4], "big") >> 2


def freeze() -> list[dict]:
    sd = seed()
    rng = random.Random(sd)
    rows, order = [], 0
    while order < POOL:
        block = list(CELLS)
        rng.shuffle(block)
        for scene, mag in block:
            if order >= POOL:
                break
            goal = _goal(rng, mag)
            pair = f"k4-panda-dev40-{order:04d}"
            for arm, margin in ARMS:
                for rep in range(REPEATS if arm == "primary" else 1):
                    label = margin_label(margin)
                    row = {"schema": "timeopt_native_source_query_v1", "protocol": PROTOCOL, "cohort": COHORT,
                           "arm": arm, "repeat": rep, "pool_order": order, "cohort_k_eligible": K, "payload": PAYLOAD,
                           "robot": "panda", "scene_id": scene, "scene_stratum": scene, "magnitude_stratum": mag,
                           "seed": sd, "query_index": order, "pair_id": pair, "solver_condition": label,
                           "planning_margin": margin, "source_id": f"{pair}-{label}-r{rep}",
                           "goal_delta_rad": [g - h for g, h in zip(goal, HOME)],
                           "start_joint_positions_rad": list(HOME), "goal_joint_positions_rad": goal,
                           "query_resolution": "FROZEN_EXPLICIT_CSPACE",
                           "planner": "pinned_curobo_motion_planner.plan_cspace",
                           "timeopt": "pinned_curobo.TrajectoryTimeOpt",
                           "source_geometry_contract": ("selected planner-integrated TimeOpt output: cloned effective cubic "
                                                        "geometry, unchanged 32 timing controls, physical actuator limits")}
                    row["source_query_sha256"] = _canonical_hash(row)
                    if not verify_query(row):
                        raise AssertionError("failed to freeze query")
                    rows.append(row)
            order += 1
    return rows


def check(rows: list[dict]) -> dict:
    per_pair = {}
    for r in rows:
        if not verify_query(r) or r["cohort"] != COHORT:
            raise ValueError("bad row " + str(r.get("source_id")))
        per_pair.setdefault(r["pair_id"], []).append(r)
    if len(per_pair) != POOL:
        raise ValueError("pool size")
    for pid, g in per_pair.items():
        if sorted((x["arm"], x["repeat"]) for x in g) != sorted(
                [("primary", k) for k in range(REPEATS)] + [("secondary", 0), ("stress", 0)]):
            raise ValueError("arm/repeat layout " + pid)
        if len({str(x["goal_joint_positions_rad"]) for x in g}) != 1:
            raise ValueError("unpaired goal " + pid)
    return {"cohort": COHORT, "pool": POOL, "k_eligible": K, "rows": len(rows), "seed": seed(), "payload": PAYLOAD}
