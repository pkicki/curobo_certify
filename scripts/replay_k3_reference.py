#!/usr/bin/env python3
"""Exact mathematical K3 proof for explicit rational/cubic pieces, NOT cuRobo."""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.boundary_reference import Span, Boundaries  # noqa: E402
from curobo_certify.exact_k3_reference import certify_k3, replay_k3_leaves  # noqa: E402


def replay(data, raw):
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("expected object with schema_version=1")
    if data.get("basis") != "local_power_ascending_u":
        raise ValueError("expected local_power_ascending_u")
    limits = data["limits"]
    spans = tuple(Span.from_values(s["left"], s["right"], s["q"], s["x"])
                  for s in data["spans"])
    targets = Boundaries.from_values(**data["targets"])
    values = (limits["velocity"], limits["acceleration"], limits["jerk"])
    result = certify_k3(spans, targets, *values,
                        max_depth=data.get("max_depth", 12),
                        max_nodes=data.get("max_nodes", 8192))
    proof_replayed = replay_k3_leaves(spans, targets, *values, result)
    if result.status == "CERTIFIED_K3_REFERENCE" and not proof_replayed:
        raise RuntimeError("reference proof replay rejected the certificate")
    return {"source_sha256": hashlib.sha256(raw).hexdigest(),
            "proof_scope": "EXACT_RATIONAL_SERIALIZED_K3_ONLY",
            "status": result.status, "native_parity": "NOT_EVALUATED",
            "full_contract_status": "UNSUPPORTED_DYNAMICS",
            "nodes_processed": result.nodes_processed,
            "proof_replay_pass": proof_replayed,
            "detail": result.detail, "violation": result.violation,
            "proof_leaves": [asdict(leaf) for leaf in result.proof_leaves]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    try:
        result = replay(json.loads(raw), raw)
    except (ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        result = {"source_sha256": hashlib.sha256(raw).hexdigest(),
                  "status": "INVALID_INPUT_OR_REPLAY", "error": str(exc),
                  "native_parity": "NOT_EVALUATED",
                  "full_contract_status": "UNSUPPORTED_DYNAMICS"}
    out = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
    else:
        print(out, end="")
    return 2 if result["status"] == "INVALID_INPUT_OR_REPLAY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
