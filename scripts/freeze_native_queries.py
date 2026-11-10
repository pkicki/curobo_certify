#!/usr/bin/env python3
"""Write the preregistered deterministic pilot queries before any solving."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from curobo_certify.query_pipeline import PILOT_SEED, pilot_queries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/native_pilot_queries.json"))
    args = parser.parse_args()
    rows = pilot_queries(PILOT_SEED)
    payload = json.dumps(rows, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    print(json.dumps({"query_count": len(rows), "seed": PILOT_SEED,
                      "manifest": str(args.out),
                      "manifest_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      "by_robot": {name: sum(row["robot"] == name for row in rows)
                                   for name in ("panda", "ur5e")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
