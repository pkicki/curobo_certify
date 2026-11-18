#!/usr/bin/env python3
"""Freeze the hypothetical-payload K4 population (no solve) from the frozen K4-Panda-Dev40 manifest."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import k4_payload as KP  # noqa: E402

base_f = ROOT / "docs/reports/k4/population/K4-Panda-Dev40.json"
rows = KP.freeze_population(json.loads(base_f.read_text()))
data = (json.dumps(rows, sort_keys=True, indent=2) + "\n").encode()
out = ROOT / "docs/reports/k4/population" / f"{KP.COHORT}.json"
if out.exists() and out.read_bytes() != data:
    raise SystemExit("REFUSING_TO_OVERWRITE_FROZEN")
out.write_bytes(data)
man = {"cohort": KP.COHORT, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows),
       "base_manifest_sha256": hashlib.sha256(base_f.read_bytes()).hexdigest(),
       "payloads": {n: {"label": KP.LABELS[n], **{k: [str(x) for x in v] if isinstance(v, list) else str(v)
                                                 for k, v in KP.inertial(n).items()}} for n in KP.PAYLOADS},
       "geometry": "uniform solid cylinder r=0.04 m, h=0.10 m, CoM at panda_hand (0,0,0.1034) m; collision NOT modelled",
       "rule": "first-K=40 per payload on the 0.95 arm over the frozen pool order; 0.99 for attempted queries; "
               "failures retained; native TimeOpt plans WITH the payload (runtime hook, audited read-back)",
       "frozen_before_any_payload_solve": True}
(ROOT / "docs/reports/k4/population" / f"{KP.COHORT}_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
print(man["sha256"], len(rows))
