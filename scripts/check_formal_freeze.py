#!/usr/bin/env python3
"""Refuse to run a formal cohort unless every frozen file still has its frozen SHA-256."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--freeze", type=Path, default=ROOT / "docs/reports/formal_panda/FREEZE.json")
    a = ap.parse_args()
    fz = json.loads(a.freeze.read_text())
    bad = []
    if fz["cohorts"][a.cohort]["sha256"] != a.expected_sha256:
        bad.append("cohort manifest sha differs from FREEZE")
    if hashlib.sha256(a.queries.read_bytes()).hexdigest() != a.expected_sha256:
        bad.append("query file sha differs")
    if a.cohort not in fz.get("opened_cohorts", []):
        bad.append(f"cohort {a.cohort} not opened by a recorded gate decision in FREEZE.json")
    for rel, sha in fz["frozen_files"].items():
        p = ROOT / rel if not rel.startswith("/") else Path(rel)
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            bad.append("frozen file changed: " + rel)
    if bad:
        print("FREEZE_CHECK_FAILED\n" + "\n".join(bad))
        sys.exit(3)
    print(f"FREEZE_CHECK_PASS {a.cohort} ({len(fz['frozen_files'])} files)")


if __name__ == "__main__":
    main()
