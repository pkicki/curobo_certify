#!/usr/bin/env python3
"""Run K3 proofs on serialized real planner outputs, retaining parity linkage."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curobo_certify.boundary_reference import Boundaries, check_boundaries  # noqa: E402
from curobo_certify.piecewise_cpu import certify_piecewise_k3, replay_piecewise_k3  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402
from export_frozen_pilot_models import metadata_paths  # noqa: E402


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 2, 4, 8, 12, 16])
    parser.add_argument("--max-nodes", type=int, default=8192)
    args = parser.parse_args()
    rows = []
    for meta_path in metadata_paths(args.pilot_dir):
        meta = json.loads(meta_path.read_text())
        if meta.get("planner_success") is not True:
            rows.append({"path_id": meta_path.stem, "native_linkage": "NO_NATIVE_OUTPUT",
                         "outcome": "UNKNOWN", "reason": meta.get("reason", meta.get("failure", "native failure"))})
            continue
        array_path = ROOT / meta["arrays_path"]
        raw = array_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != meta.get("arrays_sha256"):
            raise SystemExit(f"hash mismatch: {array_path}")
        with np.load(array_path, allow_pickle=False) as a:
            spans = extract_timeopt_spans(a["effective_geometry"].copy(), a["timing_controls"].copy())
            start = a["requested_start"][0].astype(np.float64)
            end = a["requested_goal"][0].astype(np.float64)
            # Conservative symmetric limit derived from source lower/upper limits.
            limits = [np.minimum(np.abs(a[key][0]), np.abs(a[key][1])).astype(float).tolist()
                      for key in ("velocity_limits", "acceleration_limits", "jerk_limits")]
            duration = float(a["native_duration"][0])
            sampled_duration = float(a["diagnostic_8193_times"][-1])
            native_table_duration = float(a["native_timing_time_grid"][-1])
            endpoints_equal = (spans[0].at("q", 0) == tuple(Fraction.from_float(float(v)) for v in start)
                               and spans[-1].at("q", 1) == tuple(Fraction.from_float(float(v)) for v in end))
        targets = Boundaries.from_values(
            q_start=start.tolist(), q_end=end.tolist(),
            x_start=spans[0].at("x", 0)[0], x_end=spans[-1].at("x", 1)[0],
            strict_rest=True,
        )
        budgets = []
        boundary_audit = check_boundaries(spans, targets)
        boundary_counts = {}
        for check in boundary_audit.checks:
            boundary_counts[check.name] = check.status
        for depth in args.depths:
            result = certify_piecewise_k3(spans, targets, *limits, max_depth=depth,
                                          max_nodes=args.max_nodes)
            replay_ok, replay_detail = replay_piecewise_k3(spans, targets, *limits, result)
            family_records = {name: [] for name in
                              ("positive_x", "velocity", "acceleration_positive",
                               "acceleration_negative", "jerk")}
            leaf_count = 0
            for span_row in result.span_results:
                for joint_result in span_row:
                    for leaf in joint_result.leaves:
                        leaf_count += 1
                        family_records["positive_x"].append(leaf.x_lower)
                        for family in ("velocity", "acceleration_positive",
                                       "acceleration_negative", "jerk"):
                            if family in leaf.residual_upper:
                                family_records[family].append(leaf.residual_upper[family])
            family_accounting = {}
            for family, bounds in family_records.items():
                if family == "positive_x":
                    family_accounting[family] = {
                        "leaf_checks": len(bounds), "minimum_lower_bound": min(bounds) if bounds else None,
                        "meaning": "x Bernstein positivity lower bound; each leaf requires nonnegative coefficients and moving-interior endpoint positivity"}
                else:
                    max_upper = max(bounds) if bounds else None
                    family_accounting[family] = {
                        "leaf_checks": len(bounds), "maximum_residual_upper": max_upper,
                        "margin_to_zero": -max_upper if max_upper is not None else None}
            budgets.append({
                "max_depth": depth, "k3_status": result.k3_status,
                "full_contract_status": result.full_contract_status,
                "boundary_status": result.boundary_status,
                "reason": result.reason, "processed_nodes": result.processed_nodes,
                "proof_leaves": sum(len(x.leaves) for row in result.span_results for x in row),
                "accounting_convention": "nodes are per (common span, joint) certifier call; each accepted leaf checks positivity plus velocity, positive acceleration, negative acceleration, jerk residuals. Boundary/regularity checks are exact Fraction obligations once per path.",
                "leaf_count_by_family": family_accounting,
                "boundary_obligations": boundary_counts,
                "replay_pass": replay_ok, "replay_detail": replay_detail,
            })
        rows.append({
            "path_id": meta_path.stem, "evidence_kind": meta.get("evidence_kind"),
            "native_success": True, "native_parity_status": meta.get("native_adapter_parity", {}).get("status", "UNKNOWN"),
            "native_linkage": "BLOCKED_PARITY_FAIL" if meta.get("native_adapter_parity", {}).get("status") != "PASS" else "PASS",
            "model_evidence_only": True, "arrays_sha256": sha, "duration_s": duration,
            "duration_matches_native_time_table_exactly": bool(
                np.float32(duration) == np.float32(native_table_duration)),
            "duration_matches_native_float32_endpoint_exactly": bool(
                np.float32(duration) == np.float32(sampled_duration)),
            "requested_endpoint_positions_match_exactly": endpoints_equal,
            "serialized_input_hash_verified": True,
            "span_count": len(spans), "strict_rest_requested": True,
            "limits_source": "exported native robot limit arrays; symmetric minimum absolute side",
            "budgets": budgets,
        })
    output = {"evidence_kind": "real_native_pilot_serialized_model_k3_diagnostics",
              "campaign_measurements": False,
              "warning": "Proofs concern exact serialized spline model only; native trajectory linkage fails until adapter parity passes. Torque/rate remain unsupported.",
              "depths": args.depths, "max_nodes": args.max_nodes, "paths": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"paths": len(rows), "native_successes": sum(r.get("native_success") is True for r in rows),
                      "model_evidence_only": True, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
