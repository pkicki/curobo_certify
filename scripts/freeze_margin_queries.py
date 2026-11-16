#!/usr/bin/env python3
"""Freeze matched multi-margin Panda planning queries BEFORE running cuRobo.

No GPU required. Each original source query is replicated with identical
start, goal and scene, changing only the registered TimeOpt limit scales.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.margin_campaign import DEFAULT_SCALES, freeze, check_manifest, dump_manifest, digest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--per-cell", type=int, default=3,
                   help="source queries per scene (4) x magnitude (3) stratum")
    p.add_argument("--margins", nargs="+", default=list(DEFAULT_SCALES),
                   help="registered limit_scale = jerk_limit_scale margins")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    rows = freeze(a.seed, a.per_cell, a.margins)
    data = dump_manifest(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    side = {"schema": "matched_timeopt_margin_queries_v1",
            "experiment": "PLANNING_MARGIN_DURATION_CERTIFIABILITY_EXPLORATORY",
            **check_manifest(rows), "seed": a.seed,
            "query_file_sha256": digest(data),
            "physical_limits_unchanged": True,
            "only_timeopt_scales_changed": ["limit_scale", "jerk_limit_scale"],
            "warning": "No actual trajectory has been solved; not Dev40/Holdout100/Confirm300"}
    Path(str(a.out) + ".manifest.json").write_text(
        json.dumps(side, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(side, sort_keys=True))


if __name__ == "__main__":
    main()
