#!/usr/bin/env python3
"""Freeze the amended Panda-only formal cohorts BEFORE any solve (no GPU).

Writes, per cohort, an ordered oversubscribed source-query pool (primary 0.95 + secondary
0.99 rows per source query) and a leakage audit against every earlier exploratory study.
Refuses to overwrite an existing frozen file whose bytes differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import formal_cohorts as FC  # noqa: E402

PRIOR = {
    "pilot_seed20261110": "runs/native_pilot_queries.json",
    "pilot_initial_unresolved": "runs/native_pilot_queries_initial_unresolved.json",
    "exploratory_seed20261111": "runs/compiled_hull_panda/exploratory_queries_seed20261111.json",
    "hardcase_seed20261116": "runs/hardcase_seed20261116/queries.json",
    "margins_seed20261117": "runs/margins/queries.json",
}


def write_once(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() != data:
        raise SystemExit(f"REFUSING_TO_OVERWRITE_FROZEN {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs/reports/formal_panda/queries")
    a = ap.parse_args()
    prior = {}
    for name, rel in PRIOR.items():
        rows = json.loads((ROOT / rel).read_text())
        prior[name] = [r for r in rows if r.get("robot") == "panda" and r.get("goal_joint_positions_rad")]
    cohorts = {c: FC.freeze_cohort(c) for c in FC.COHORTS}
    leak = FC.leakage_audit(cohorts, prior)
    if not leak["pass"]:
        raise SystemExit("LEAKAGE_AUDIT_FAILED " + json.dumps(leak))
    side = {"protocol": FC.PROTOCOL, "population_flag": FC.POPULATION_FLAG,
            "original_dual_robot_protocol": FC.ORIGINAL_PROTOCOL_STATUS,
            "primary_margin": FC.PRIMARY, "secondary_margin": FC.SECONDARY,
            "prior_query_files_sha256": {n: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for n, p in PRIOR.items()},
            "leakage_audit": leak, "cohorts": {}}
    for c, rows in cohorts.items():
        data = FC.dump(rows)
        path = a.out_dir / f"{c}.json"
        write_once(path, data)
        side["cohorts"][c] = {**FC.check_cohort(rows, c), "seed": FC.cohort_seed(c), "file": path.name,
                              "sha256": hashlib.sha256(data).hexdigest()}
    write_once(a.out_dir / "population_manifest.json", (json.dumps(side, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({c: (v["sha256"], v["pool"], v["k_eligible"]) for c, v in side["cohorts"].items()}, indent=1))


if __name__ == "__main__":
    main()
