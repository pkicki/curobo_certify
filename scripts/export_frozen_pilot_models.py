#!/usr/bin/env python3
"""Export exact serialized q/x polynomials from retained native pilot arrays.

This uses the declared control-polygon spline extraction; no samples are fit.
Exports are mathematical-model inputs, not native-linked certificates.
"""
import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


def export_one(meta_path: Path, out_dir: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    if meta.get("planner_success") is not True:
        return {"path_id": meta_path.stem, "status": "NO_NATIVE_OUTPUT", "source_metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest()}
    arrays_path = (ROOT / meta["arrays_path"]).resolve()
    raw = arrays_path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != meta.get("arrays_sha256"):
        raise ValueError(f"source array hash mismatch: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        spans = extract_timeopt_spans(arrays["effective_geometry"].copy(), arrays["timing_controls"].copy())
        limits = {
            name: np.minimum(np.abs(arrays[key][0]), np.abs(arrays[key][1])).astype(float).tolist()
            for name, key in (("velocity", "velocity_limits"), ("acceleration", "acceleration_limits"), ("jerk", "jerk_limits"))
        }
        targets = {
            "q_start": [str(Fraction.from_float(float(x))) for x in arrays["requested_start"][0]],
            "q_end": [str(Fraction.from_float(float(x))) for x in arrays["requested_goal"][0]],
            "x_start": str(spans[0].at("x", 0)[0]),
            "x_end": str(spans[-1].at("x", 1)[0]),
        }
        span_rows = [{"left": str(s.left), "right": str(s.right),
                      "q": [[str(c) for c in joint] for joint in s.q],
                      "x": [str(c) for c in s.x]} for s in spans]
        duration = float(arrays["native_duration"][0])
        timing_sha = hashlib.sha256(arrays["timing_controls"].tobytes()).hexdigest()
        geometry_sha = hashlib.sha256(arrays["effective_geometry"].tobytes()).hexdigest()
    model = {"schema_version": 1, "basis": "local_power_ascending_u", "spans": span_rows,
             "targets": targets, "limits": limits,
             "provenance": {"path_id": meta_path.stem, "evidence_kind": meta.get("evidence_kind"),
                            "source_arrays_sha256": source_sha, "source_metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                            "effective_geometry_sha256": geometry_sha, "timing_controls_sha256": timing_sha,
                            "upstream_sha": meta.get("upstream_sha"), "native_duration_seconds": duration,
                            "strict_rest_requested": True,
                            "conversion": "exact-rational extraction from cloned effective cubic path controls and 32 timing controls; no fitting",
                            "native_linkage": "BLOCKED_PARITY_FAIL"}}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{meta_path.stem}.json"
    encoded = (json.dumps(model, sort_keys=True, indent=2) + "\n").encode()
    out.write_bytes(encoded)
    return {"path_id": meta_path.stem, "status": "EXPORTED_MODEL_ONLY", "model_path": str(out),
            "model_sha256": hashlib.sha256(encoded).hexdigest(), "source_arrays_sha256": source_sha,
            "span_count": len(spans)}


def metadata_paths(pilot_dir: Path):
    """Select per-query Panda records, including separately named exploratory cohorts."""
    for path in sorted(pilot_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        if not (path.name.startswith("pilot-panda-") or path.name.startswith("explore-panda-")):
            continue
        yield path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    rows = [export_one(path, a.out_dir) for path in metadata_paths(a.pilot_dir)]
    manifest = {"schema_version": 1, "scope": "EXACT_SERIALIZED_MATHEMATICAL_MODEL_ONLY", "campaign_eligible": False, "paths": rows}
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "exported": sum(x["status"] == "EXPORTED_MODEL_ONLY" for x in rows), "manifest": str(a.manifest)}, indent=2))


if __name__ == "__main__":
    main()
