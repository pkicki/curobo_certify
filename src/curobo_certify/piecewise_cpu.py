"""Boundary-complete multi-span K3 wrapper over the outward CPU checker.

Inputs are local-power polynomials interpreted exactly as their Fraction / IEEE
binary64 values. The boundary oracle is exact rational; residuals and leaves use
the directed binary64 Bernstein implementation. Torque and torque-rate remain
unsupported, so a K3 pass is never a full dynamics certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from fractions import Fraction
from typing import Sequence

from . import exact_k3_reference as exact
from .bernstein import power_to_bernstein
from .boundary_reference import Boundaries, Span, check_boundaries
from .certify_cpu import SpanResult, certify_span
from .proof_check import verify_span, verify_violation


@dataclass
class PiecewiseResult:
    k3_status: str
    full_contract_status: str
    boundary_status: str
    reason: str
    processed_nodes: int
    span_results: list[list[SpanResult]]
    boundary_evidence: list[dict] = field(default_factory=list)


def _bernstein(span: Span):
    return [power_to_bernstein(poly) for poly in span.q], power_to_bernstein(span.x)


def certify_piecewise_k3(
    spans: Sequence[Span],
    targets: Boundaries,
    vlim,
    alim,
    jlim,
    *,
    max_depth: int = 12,
    max_nodes: int = 8192,
) -> PiecewiseResult:
    """Certify exact boundaries plus all per-joint outward K3 span proofs.

    The node limit is a path-global count across every span/joint call. Inputs
    must already use exact native coefficient interpretation and common knots.
    """
    boundary = check_boundaries(spans, targets)
    if boundary.status != "PASS":
        status = boundary.status if boundary.status in ("VIOLATED", "INVALID_INPUT") else "UNKNOWN_BOUNDARY"
        return PiecewiseResult(status, "UNSUPPORTED_DYNAMICS", boundary.status,
                               "boundary/regularity obligation did not pass", 0, [],
                               [{"name": c.name, "status": c.status, "detail": c.detail}
                                for c in boundary.checks if c.status != "PASS"])
    if not isinstance(max_depth, int) or not isinstance(max_nodes, int) or max_depth < 0 or max_nodes < 1:
        return PiecewiseResult("INVALID_INPUT", "UNSUPPORTED_DYNAMICS", "PASS",
                               "invalid proof budget", 0, [])
    if not spans or not vlim or len(vlim) != len(alim) or len(vlim) != len(jlim):
        return PiecewiseResult("INVALID_INPUT", "UNSUPPORTED_DYNAMICS", "PASS",
                               "empty/mismatched joint limits", 0, [])

    all_results: list[list[SpanResult]] = []
    processed = 0
    for span in spans:
        q_bern, x_bern = _bernstein(span)
        if len(q_bern) != len(vlim):
            return PiecewiseResult("INVALID_INPUT", "UNSUPPORTED_DYNAMICS", "PASS",
                                   "joint dimension differs from limits", processed, all_results)
        row = []
        for joint, q in enumerate(q_bern):
            remaining = max_nodes - processed
            if remaining <= 0:
                return PiecewiseResult("UNKNOWN_NODES", "UNSUPPORTED_DYNAMICS", "PASS",
                                       "path-global proof node budget exhausted", processed, all_results)
            result = certify_span(
                q, x_bern, (vlim[joint], alim[joint], jlim[joint]),
                max_depth=max_depth, max_nodes=remaining,
                span_length=span.right - span.left,
            )
            row.append(result)
            processed += result.processed_nodes
            if result.k3_status == "VIOLATED":
                witness_ok, witness_detail = verify_violation(
                    q, x_bern, (vlim[joint], alim[joint], jlim[joint]), result,
                    span_length=span.right - span.left,
                )
                if not witness_ok:
                    all_results.append(row)
                    return PiecewiseResult("NUMERICAL_ERROR", "UNSUPPORTED_DYNAMICS", "PASS",
                                           "violation witness replay failed: " + witness_detail,
                                           processed, all_results)
                all_results.append(row)
                return PiecewiseResult("VIOLATED", "UNSUPPORTED_DYNAMICS", "PASS",
                                       f"rigorous residual witness on span joint {joint}: {witness_detail}",
                                       processed, all_results)
            if result.k3_status not in ("UNKNOWN_BOUNDARY", "CERTIFIED"):
                all_results.append(row)
                return PiecewiseResult(result.k3_status, "UNSUPPORTED_DYNAMICS", "PASS",
                                       result.reason, processed, all_results)
            ok, why = verify_span(q, x_bern, (vlim[joint], alim[joint], jlim[joint]), result,
                                  span_length=span.right - span.left)
            if not ok:
                all_results.append(row)
                return PiecewiseResult("NUMERICAL_ERROR", "UNSUPPORTED_DYNAMICS", "PASS",
                                       f"internal proof replay failed: {why}", processed, all_results)
        all_results.append(row)

    return PiecewiseResult("CERTIFIED_K3", "UNSUPPORTED_DYNAMICS", "PASS",
                           "exact boundary and independent outward-leaf replay passed; dynamics unsupported",
                           processed, all_results)


def replay_piecewise_k3(spans: Sequence[Span], targets: Boundaries, vlim, alim, jlim,
                        result: PiecewiseResult) -> tuple[bool, str]:
    """Independently replay all leaf partitions after rechecking exact boundaries."""
    if result.k3_status != "CERTIFIED_K3" or result.full_contract_status != "UNSUPPORTED_DYNAMICS":
        return False, "result does not claim the supported K3-only scope"
    boundary = check_boundaries(spans, targets)
    if boundary.status != "PASS" or result.boundary_status != "PASS":
        return False, "exact boundary/regularity replay failed"
    if len(result.span_results) != len(spans):
        return False, "span proof count mismatch"
    expected_joints = len(vlim)
    for span, results in zip(spans, result.span_results):
        if len(results) != expected_joints:
            return False, "joint proof count mismatch"
        q_bern, x_bern = _bernstein(span)
        for joint, proof in enumerate(results):
            ok, why = verify_span(q_bern[joint], x_bern,
                                  (vlim[joint], alim[joint], jlim[joint]), proof,
                                  span_length=span.right - span.left)
            if not ok:
                return False, why
    return True, "all exact boundaries and outward-rounded K3 leaves replayed"


def exact_oracle_status(spans: Sequence[Span], targets: Boundaries, vlim, alim, jlim,
                        *, max_depth: int = 12, max_nodes: int = 8192) -> str:
    """Expose the Fraction backend for differential tests and debugging only."""
    return exact.certify_k3(spans, targets, vlim, alim, jlim,
                            max_depth=max_depth, max_nodes=max_nodes).status
