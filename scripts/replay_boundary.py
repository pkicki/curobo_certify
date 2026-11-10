#!/usr/bin/env python3
"""Replay a serialized boundary-only proof from explicit local-power spans.

This is NOT a TimeOpt adapter and it cannot establish a full certificate.
Input numbers are interpreted as exact JSON integer/binary float values, or
rational strings (e.g. "1/3"). No spline interpolation or fitting occurs.
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.boundary_reference import Span, Boundaries, check_boundaries  # noqa: E402


def replay(data: dict, sha256: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    if data.get("schema_version") != 1:
        raise ValueError("expected boundary replay schema_version=1")
    if data.get("basis") != "local_power_ascending_u":
        raise ValueError("expected local_power_ascending_u coefficients")
    if not isinstance(data.get("spans"), list) or not isinstance(data.get("targets"), dict):
        raise ValueError("spans list and targets object are required")
    spans = [Span.from_values(s["left"], s["right"], s["q"], s["x"])
             for s in data["spans"]]
    targets = Boundaries.from_values(**data["targets"])
    result = check_boundaries(spans, targets)
    return {"source_sha256": sha256, "proof_scope": "BOUNDARY_ONLY",
            "status": "BOUNDARY_" + result.status,
            "checks": [asdict(c) for c in result.checks],
            "full_k3_status": "NOT_EVALUATED", "full_contract_status": "NOT_EVALUATED"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    try:
        data = json.loads(source)
        result = replay(data, digest)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        result = {"source_sha256": digest, "proof_scope": "BOUNDARY_ONLY",
                  "status": "BOUNDARY_INVALID_INPUT", "error": str(exc),
                  "full_k3_status": "NOT_EVALUATED", "full_contract_status": "NOT_EVALUATED"}
    serialized = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 2 if result["status"] == "BOUNDARY_INVALID_INPUT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
