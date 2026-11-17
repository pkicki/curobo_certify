"""Run the randomized K4 P1 soundness campaign and write evidence JSON.

  PYTHONPATH=src:. python independent_checks/k4/run_review.py \
      --out docs/reports/k4_p1_independent_review/campaign_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from independent_checks.k4.campaign import run_campaign  # noqa: E402

REVIEWED = ["src/curobo_certify/k4_reference.py", "src/curobo_certify/k4_affine.py",
            "tests/test_k4_reference.py", "tests/test_k4_affine.py",
            "scripts/run_k4_reference_pilot.py", "docs/K4_CERTIFIED_TORQUE_RESEARCH_PLAN_20261117.md"]
ORACLE = ["independent_checks/k4/oracle.py", "independent_checks/k4/campaign.py",
          "independent_checks/k4/lagrange_sympy.py", "independent_checks/k4/run_review.py"]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404])
    ap.add_argument("--cases", type=int, default=150)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--max-nodes", type=int, default=600)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("refusing to overwrite")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    runs = []
    t0 = time.time()
    for s in a.seeds:
        r = run_campaign(s, a.cases, max_depth=a.max_depth, max_nodes=a.max_nodes, dense=64, extra=3,
                         affine_rate=0.4)
        runs.append(r)
        print(s, r["status"], "counterexamples", len(r["counterexamples"]), flush=True)
    total = {"cases": sum(r["cases"] for r in runs),
             "leaf_checks": sum(r["leaf_checks"] + r["affine_leaf_checks"] for r in runs),
             "counterexamples": sum(len(r["counterexamples"]) for r in runs),
             "ambiguous": sum(r["ambiguous"] for r in runs)}
    out = {"scope": "INDEPENDENT_REVIEW_SAMPLED_DIAGNOSTIC_AGAINST_RIGOROUS_POINT_ORACLE",
           "note": "Dense oracle sampling is a falsification test, not a proof of the reviewed bounds.",
           "git_head": head, "python": platform.python_version(),
           "reviewed_sha256": {p: sha(p) for p in REVIEWED},
           "oracle_sha256": {p: sha(p) for p in ORACLE},
           "params": vars(a) | {"out": str(a.out)}, "wall_s": round(time.time() - t0, 1),
           "total": total, "runs": runs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
