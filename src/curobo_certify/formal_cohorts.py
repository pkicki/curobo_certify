"""Amended Panda-only formal qualification population (AMENDED_POPULATION_PANDA_ONLY).

The historical protocol (docs/CAMPAIGN_PLAN.md section 7) registered Dev40 / Holdout100 /
Confirm300 as 20/50/150 *paths* per robot for Panda + UR5e. UR5e remains
BLOCKED_UR5E_MODEL_LIMITS (no supported pinned cuRobo model, no sourced acceleration/jerk
limits), so this module defines a SEPARATELY NAMED Panda-only protocol:
Panda-Dev40 / Panda-Holdout100 / Panda-Confirm300. It is never the original dual-robot
qualification.

Sample unit (preregistered): K eligible primary-arm native successes per cohort, chosen by a
deterministic first-K rule over a frozen, oversubscribed, ordered source-query pool. Every
attempted source query (including native failures) is retained in the ledger; failures are
never replaced out of order, and if the pool is exhausted the shortfall is reported.

Arms: primary = unmodified cuRobo TimeOpt (limit_scale = jerk_limit_scale = 0.95, unpatched
constructor); secondary = 0.99 on the same source queries (separately reported, never pooled).

Pure data; no GPU. Only the frozen generator may instantiate cuRobo.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Iterable

from .margin_campaign import HOME, LIMITS, MAGNITUDES, SCENES, UPSTREAM_SHA, margin_label
from .query_pipeline import _canonical_hash, verify_query

PROTOCOL = "panda_only_formal_qualification_v1"
POPULATION_FLAG = "AMENDED_POPULATION_PANDA_ONLY"
ORIGINAL_PROTOCOL_STATUS = "BLOCKED_UR5E_MODEL_LIMITS"
PRIMARY = "0.95"
SECONDARY = "0.99"
# cohort -> (K eligible primary successes, oversubscribed pool size)
COHORTS = {"Panda-Dev40": (40, 70), "Panda-Holdout100": (100, 175), "Panda-Confirm300": (300, 525)}
CELLS = tuple((scene, mag) for scene in SCENES for mag in MAGNITUDES)  # 12 strata
SEED_SALT = "curobo_certify/panda_formal_v1/"


def cohort_seed(cohort: str) -> int:
    """Independent per-cohort seed derived by SHA-256 (distinct from all exploratory seeds)."""
    if cohort not in COHORTS:
        raise ValueError(cohort)
    # 31-bit: the pinned planner forwards seed+query_index to RNGs that may require < 2**32.
    return int.from_bytes(hashlib.sha256((SEED_SALT + cohort).encode()).digest()[:4], "big") >> 2


def _goal(rng: random.Random, magnitude: str) -> list[float]:
    lo, hi = MAGNITUDES[magnitude]
    goal = []
    for j, home in enumerate(HOME):
        lower, upper = LIMITS[j]
        center, radius = (lower + upper) / 2, 0.45 * (upper - lower)
        position = home + rng.choice((-1, 1)) * rng.uniform(lo, hi)
        goal.append(min(max(position, center - radius), center + radius))
    return goal


def freeze_cohort(cohort: str) -> list[dict]:
    """Ordered pool: blocks of 12 = one query per (scene, magnitude) stratum, block order seeded."""
    k, pool = COHORTS[cohort]
    seed = cohort_seed(cohort)
    rng = random.Random(seed)
    rows: list[dict] = []
    order = 0
    while order < pool:
        block = list(CELLS)
        rng.shuffle(block)
        for scene, mag in block:
            if order >= pool:
                break
            goal = _goal(rng, mag)
            pair = f"{cohort.lower()}-{order:04d}"
            for margin in (PRIMARY, SECONDARY):
                label = margin_label(margin)
                row = {
                    "schema": "timeopt_native_source_query_v1",
                    "protocol": PROTOCOL,
                    "population_flag": POPULATION_FLAG,
                    "cohort": cohort,
                    "arm": "primary" if margin == PRIMARY else "secondary",
                    "pool_order": order,
                    "cohort_k_eligible": k,
                    "robot": "panda",
                    "scene_id": scene,
                    "scene_stratum": scene,
                    "magnitude_stratum": mag,
                    "seed": seed,
                    "query_index": order,
                    "pair_id": pair,
                    "solver_condition": label,
                    "planning_margin": margin,
                    "source_id": f"{pair}-{label}",
                    "goal_delta_rad": [g - h for g, h in zip(goal, HOME)],
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
                rows.append(row)
            order += 1
    return rows


def dump(rows: list[dict]) -> bytes:
    return (json.dumps(rows, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _goals(rows: Iterable[dict]) -> dict[str, tuple[float, ...]]:
    out = {}
    for r in rows:
        out[r.get("pair_id") or r["source_id"]] = tuple(float(v) for v in r["goal_joint_positions_rad"])
    return out


def leakage_audit(cohorts: dict[str, list[dict]], prior: dict[str, list[dict]]) -> dict:
    """No shared seed/source id/pair id/goal with any prior study or between cohorts.

    Start (planner HOME) and the four scenes are fixed strata shared by every study; the
    independent sampling unit is the goal configuration, so leakage is checked on goals."""
    prior_seeds = {r.get("seed") for rows in prior.values() for r in rows}
    prior_ids = {r.get("source_id") for rows in prior.values() for r in rows}
    prior_goals = [g for rows in prior.values() for g in _goals(rows).values()]
    report = {"prior_studies": {k: len(v) for k, v in prior.items()}, "cohorts": {}}
    ok = True
    all_goals = {name: list(_goals(rows).values()) for name, rows in cohorts.items()}

    def dmin(a, b):
        d = min((max(abs(x - y) for x, y in zip(g, p)) for g in a for p in b), default=None)
        return d

    for name, rows in cohorts.items():
        goals = all_goals[name]
        seeds = {r["seed"] for r in rows}
        ids = {r["source_id"] for r in rows}
        others = [g for other, gs in all_goals.items() if other != name for g in gs]
        within = [max(abs(x - y) for x, y in zip(goals[i], goals[j])) for i in range(len(goals)) for j in range(i)]
        c = {"seed": sorted(seeds), "seed_reused": bool(seeds & prior_seeds), "source_id_reused": bool(ids & prior_ids),
             "min_linf_goal_distance_to_prior_rad": dmin(goals, prior_goals),
             "min_linf_goal_distance_to_other_cohorts_rad": dmin(goals, others),
             "min_linf_goal_distance_within_cohort_rad": min(within) if within else None}
        c["pass"] = (not c["seed_reused"] and not c["source_id_reused"]
                     and all(v is None or v > 1e-6 for k, v in c.items() if k.startswith("min_")))
        ok &= c["pass"]
        report["cohorts"][name] = c
    report["pass"] = bool(ok)
    report["shared_fixed_strata"] = {"start": "pinned planner HOME (all studies)", "scenes": list(SCENES)}
    return report


def check_cohort(rows: list[dict], cohort: str) -> dict:
    k, pool = COHORTS[cohort]
    if len(rows) != 2 * pool:
        raise ValueError("pool size mismatch")
    pairs: dict[str, list[dict]] = {}
    for r in rows:
        if not verify_query(r) or r["cohort"] != cohort or r["population_flag"] != POPULATION_FLAG:
            raise ValueError("bad cohort row " + str(r.get("source_id")))
        pairs.setdefault(r["pair_id"], []).append(r)
    for pid, group in pairs.items():
        if sorted(g["planning_margin"] for g in group) != [PRIMARY, SECONDARY]:
            raise ValueError("arm mismatch " + pid)
        if len({json.dumps(g["goal_joint_positions_rad"]) for g in group}) != 1:
            raise ValueError("unpaired goal " + pid)
    orders = sorted({r["pool_order"] for r in rows})
    if orders != list(range(pool)):
        raise ValueError("pool order gap")
    strata = {}
    for r in rows:
        if r["arm"] == "primary":
            strata[f"{r['scene_stratum']}/{r['magnitude_stratum']}"] = strata.get(
                f"{r['scene_stratum']}/{r['magnitude_stratum']}", 0) + 1
    return {"cohort": cohort, "k_eligible": k, "pool": pool, "rows": len(rows), "pairs": len(pairs),
            "strata_counts_in_pool": strata, "upstream_sha": UPSTREAM_SHA}


def first_k_selection(primary_status_in_order: list[tuple[int, bool]], k: int) -> dict:
    """Deterministic first-K rule: solve the primary arm in pool order; stop after the K-th
    success. Returns attempted orders, selected (eligible) orders and shortfall."""
    attempted, selected = [], []
    last = -1
    for order, ok in sorted(primary_status_in_order):
        if order != last + 1:
            raise ValueError("primary arm not attempted in pool order (gap at %d)" % order)
        last = order
        attempted.append(order)
        if ok:
            selected.append(order)
            if len(selected) == k:
                break
    return {"attempted": attempted, "selected": selected, "shortfall": max(0, k - len(selected))}
