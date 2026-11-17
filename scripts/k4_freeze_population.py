#!/usr/bin/env python3
"""Freeze the K4 exploratory Panda population (no solve), with leakage audit against every prior
Panda query file, including the K3 formal pools. Refuses to overwrite a different frozen file."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import k4_population as P  # noqa: E402
from curobo_certify.formal_cohorts import leakage_audit  # noqa: E402

PRIOR = {
    "pilot_seed20261110": "runs/native_pilot_queries.json",
    "exploratory_seed20261111": "runs/compiled_hull_panda/exploratory_queries_seed20261111.json",
    "hardcase_seed20261116": "runs/hardcase_seed20261116/queries.json",
    "margins_seed20261117": "runs/margins/queries.json",
    "k3_formal_Panda-Dev40": "docs/reports/formal_panda/queries/Panda-Dev40.json",
    "k3_formal_Panda-Holdout100": "docs/reports/formal_panda/queries/Panda-Holdout100.json",
    "k3_formal_Panda-Confirm300": "docs/reports/formal_panda/queries/Panda-Confirm300.json",
}
OUT = ROOT / "docs/reports/k4/population"


def main():
    prior = {}
    for n, rel in PRIOR.items():
        rows = json.loads((ROOT / rel).read_text())
        prior[n] = [r for r in rows if r.get("robot") == "panda" and r.get("goal_joint_positions_rad")]
    rows = P.freeze()
    info = P.check(rows)
    leak = leakage_audit({P.COHORT: rows}, prior)
    if not leak["pass"]:
        raise SystemExit("LEAKAGE " + json.dumps(leak))
    data = P.dump(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{P.COHORT}.json"
    if f.exists() and f.read_bytes() != data:
        raise SystemExit("REFUSING_TO_OVERWRITE_FROZEN")
    f.write_bytes(data)
    man = {**info, "protocol": P.PROTOCOL, "sha256": hashlib.sha256(data).hexdigest(), "file": f.name,
           "arms": P.ARMS, "repeats_primary": P.REPEATS, "leakage_audit": leak,
           "prior_files_sha256": {n: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for n, p in PRIOR.items()},
           "blocked": {"payload_3kg": "BLOCKED_PAYLOAD", "payload_5kg": "BLOCKED_PAYLOAD"}}
    (OUT / "population_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(man["sha256"], info, leak["cohorts"][P.COHORT])


if __name__ == "__main__":
    main()
