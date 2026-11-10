#!/usr/bin/env python3
"""Run boundary-complete outward K3 CPU certification on explicit polynomials."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curobo_certify.boundary_reference import Boundaries, Span  # noqa: E402
from curobo_certify.piecewise_cpu import certify_piecewise_k3, replay_piecewise_k3  # noqa: E402


def run(data: dict, raw: bytes) -> dict:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("expected schema_version=1 object")
    if data.get("basis") != "local_power_ascending_u":
        raise ValueError("expected local_power_ascending_u basis")
    spans = tuple(Span.from_values(s["left"], s["right"], s["q"], s["x"])
                  for s in data["spans"])
    targets = Boundaries.from_values(**data["targets"])
    limits = data["limits"]
    values = (limits["velocity"], limits["acceleration"], limits["jerk"])
    result = certify_piecewise_k3(
        spans, targets, *values,
        max_depth=data.get("max_depth", 12),
        max_nodes=data.get("max_nodes", 8192),
    )
    replay, replay_detail = replay_piecewise_k3(spans, targets, *values, result)
    if result.k3_status == "CERTIFIED_K3" and not replay:
        raise RuntimeError("separate outward proof checker rejected claimed K3 certificate")
    return {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "proof_scope": "OUTWARD_CPU_K3_FOR_EXACT_SERIALIZED_POLYNOMIALS",
        "native_parity": "NOT_EVALUATED",
        "k3_status": result.k3_status,
        "full_contract_status": result.full_contract_status,
        "boundary_status": result.boundary_status,
        "proof_replay_pass": replay,
        "proof_replay_detail": replay_detail,
        "processed_nodes": result.processed_nodes,
        "spans": [[asdict(proof) for proof in row] for row in result.span_results],
        "boundary_evidence": result.boundary_evidence,
        "reason": result.reason,
        "data_class": "ANALYTIC_FIXTURE_OR_EXPLICIT_INPUT_NOT_BENCHMARK_DATA",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    try:
        output = run(json.loads(raw), raw)
    except (ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        output = {"source_sha256": hashlib.sha256(raw).hexdigest(),
                  "k3_status": "INVALID_INPUT", "native_parity": "NOT_EVALUATED",
                  "full_contract_status": "UNSUPPORTED_DYNAMICS", "error": str(exc)}
    rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    else:
        print(rendered, end="")
    return 2 if output["k3_status"] == "INVALID_INPUT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
