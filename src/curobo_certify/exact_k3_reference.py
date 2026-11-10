"""Exact-rational, small-scale K3 certifier for serialized cubic q(s), x(s).

This is a mathematical reference for *exactly represented* input polynomials.
It is not a native TimeOpt parity proof, an outward-float CPU backend, or a
GPU implementation. Torque and torque rate are not considered.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Sequence

from curobo_certify.boundary_reference import (Boundaries, Span, check_boundaries,
                                                derivative, rational)

ZERO = Fraction(0)
ONE = Fraction(1)


def poly_add(a, b):
    return tuple((a[i] if i < len(a) else ZERO) + (b[i] if i < len(b) else ZERO)
                 for i in range(max(len(a), len(b))))


def poly_scale(a, factor):
    return tuple(coefficient * factor for coefficient in a)


def poly_mul(a, b):
    out = [ZERO] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return tuple(out)


def bernstein_coeffs(power):
    """Exact power->Bernstein conversion on u in [0,1]."""
    n = len(power) - 1
    if n < 0:
        raise ValueError("empty polynomial")
    return tuple(sum((power[i] * Fraction(comb(k, i), comb(n, i))
                      for i in range(k + 1)), ZERO) for k in range(n + 1))


def midpoint_split(b):
    """Exact midpoint de Casteljau without changing the source polynomial."""
    levels = [tuple(b)]
    while len(levels[-1]) > 1:
        v = levels[-1]
        levels.append(tuple((v[i] + v[i + 1]) / 2 for i in range(len(v) - 1)))
    n = len(b) - 1
    return (tuple(levels[i][0] for i in range(n + 1)),
            tuple(levels[n - i][i] for i in range(n + 1)))


def build_residuals(span: Span, vlim, alim, jlim):
    """Yield (family, joint, local power residual), all <=0 to certify."""
    if not (len(vlim) == len(alim) == len(jlim) == len(span.q)):
        raise ValueError("joint limits must match the path joint dimension")
    values = [(tuple(rational(z) for z in seq)) for seq in (vlim, alim, jlim)]
    if any(z <= 0 for seq in values for z in seq):
        raise ValueError("all limits must be strictly positive")
    vlim, alim, jlim = values
    h = span.right - span.left
    x = span.x
    dx, ddx = derivative(x, 1, h), derivative(x, 2, h)
    yield ("positive_x", -1, poly_scale(x, -1))
    for joint, q in enumerate(span.q):
        dq, ddq, dddq = (derivative(q, order, h) for order in (1, 2, 3))
        v2 = poly_mul(poly_mul(dq, dq), x)
        yield ("velocity", joint, poly_add(poly_scale(v2, ONE / (vlim[joint] ** 2)), (-ONE,)))
        acc = poly_add(poly_mul(ddq, x), poly_scale(poly_mul(dq, dx), Fraction(1, 2)))
        yield ("acceleration_plus", joint, poly_add(poly_scale(acc, ONE / alim[joint]), (-ONE,)))
        yield ("acceleration_minus", joint, poly_add(poly_scale(acc, -ONE / alim[joint]), (-ONE,)))
        ell = poly_add(poly_add(poly_mul(dddq, x),
                                poly_scale(poly_mul(ddq, dx), Fraction(3, 2))),
                       poly_scale(poly_mul(dq, ddx), Fraction(1, 2)))
        j2 = poly_mul(x, poly_mul(ell, ell))
        yield ("jerk", joint, poly_add(poly_scale(j2, ONE / (jlim[joint] ** 2)), (-ONE,)))


@dataclass(frozen=True)
class Leaf:
    residual_id: str
    path: str  # Sequence of L/R midpoint children; empty means no subdivision.
    coefficients: tuple[str, ...]  # Exact Bernstein rational numbers.


@dataclass(frozen=True)
class K3Result:
    status: str  # CERTIFIED_K3_REFERENCE, VIOLATED, UNKNOWN_*, INVALID_INPUT.
    scope: str
    nodes_processed: int
    proof_leaves: tuple[Leaf, ...]
    detail: str
    violation: str | None = None


def _check_one(b0, residual_id, strict, max_depth, max_nodes, processed):
    """Return status, leaves, nodes, evidence. No approximate witnesses."""
    queue = [(tuple(b0), "", ZERO, ONE)]
    leaves = []
    while queue:
        if processed >= max_nodes:
            return "UNKNOWN_NODES", leaves, processed, f"node budget for {residual_id} exhausted"
        b, path, left, right = queue.pop(0)
        processed += 1
        lchild, _ = midpoint_split(b)
        midpoint_value = lchild[-1]
        if b[0] > 0 or b[-1] > 0 or midpoint_value > 0:
            where = "left" if b[0] > 0 else "right" if b[-1] > 0 else "mid"
            witness = b[0] if where == "left" else b[-1] if where == "right" else midpoint_value
            return "VIOLATED", leaves, processed, f"{residual_id} positive at {where} of {path or 'root'}: {witness}"
        if strict and ((left > 0 and b[0] == 0) or (right < 1 and b[-1] == 0)):
            return "UNKNOWN_STATIONARY", leaves, processed, f"interior stationary root in {residual_id}"
        if max(b) <= 0 and (not strict or min(b) < 0):
            leaves.append(Leaf(residual_id, path, tuple(map(str, b))))
            continue
        if strict and max(b) == min(b) == 0:
            return "UNKNOWN_STATIONARY", leaves, processed, f"identically stationary span in {residual_id}"
        if len(path) >= max_depth:
            return "UNKNOWN_DEPTH", leaves, processed, f"depth {max_depth} exhausted for {residual_id}"
        lc, rc = midpoint_split(b)
        middle = (left + right) / 2
        queue.extend(((lc, path + "L", left, middle), (rc, path + "R", middle, right)))
    return "CERTIFIED", leaves, processed, "all exact Bernstein upper bounds <=0"


def certify_k3(spans: Sequence[Span], targets: Boundaries, vlim, alim, jlim,
               max_depth: int = 12, max_nodes: int = 8192) -> K3Result:
    """Reference certification; asserts only exact serialized polynomial model."""
    scope = "EXACT_RATIONAL_SERIALIZED_K3_ONLY"
    if not isinstance(max_depth, int) or not isinstance(max_nodes, int) or max_depth < 0 or max_nodes <= 0:
        return K3Result("INVALID_INPUT", scope, 0, (), "invalid proof budgets")
    boundaries = check_boundaries(spans, targets)
    if boundaries.status != "PASS":
        status = "VIOLATED" if boundaries.status == "VIOLATED" else "UNKNOWN_BOUNDARY"
        if boundaries.status == "INVALID_INPUT":
            status = "INVALID_INPUT"
        return K3Result(status, scope, 0, (), "boundary/regularity sub-contract: " + boundaries.status)
    proof = []
    count = 0
    try:
        for piece, span in enumerate(spans):
            for family, joint, power in build_residuals(span, vlim, alim, jlim):
                identifier = f"span{piece}/{family}/joint{joint}"
                state, leaves, count, detail = _check_one(bernstein_coeffs(power), identifier,
                                                          family == "positive_x", max_depth, max_nodes, count)
                proof.extend(leaves)
                if state != "CERTIFIED":
                    return K3Result(state, scope, count, tuple(proof), detail,
                                    detail if state == "VIOLATED" else None)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        return K3Result("INVALID_INPUT", scope, count, tuple(proof), str(exc))
    return K3Result("CERTIFIED_K3_REFERENCE", scope, count, tuple(proof),
                    "exact boundary and all K3 polynomial residual certificates passed")


def replay_k3_leaves(spans, targets, vlim, alim, jlim, result: K3Result) -> bool:
    """Independently reconstruct every leaf and verify the complete partition."""
    if result.status != "CERTIFIED_K3_REFERENCE" or check_boundaries(spans, targets).status != "PASS":
        return False
    expected = {}
    try:
        for piece, span in enumerate(spans):
            for family, joint, power in build_residuals(span, vlim, alim, jlim):
                expected[f"span{piece}/{family}/joint{joint}"] = (bernstein_coeffs(power), family == "positive_x")
    except (ValueError, TypeError, ZeroDivisionError):
        return False
    grouped = {key: [] for key in expected}
    for leaf in result.proof_leaves:
        if leaf.residual_id not in grouped:
            return False
        grouped[leaf.residual_id].append(leaf)
    for key, (root, strict) in expected.items():
        entries = grouped[key]
        paths = [leaf.path for leaf in entries]
        if len(paths) != len(set(paths)) or not paths:
            return False
        # A prefix-free collection covers [0,1] iff dyadic leaf measures sum to 1.
        if any(path.startswith(other) for path in paths for other in paths if path != other):
            return False
        if sum((Fraction(1, 2 ** len(path)) for path in paths), ZERO) != ONE:
            return False
        for leaf in entries:
            b = root
            left, right = ZERO, ONE
            for step in leaf.path:
                lc, rc = midpoint_split(b)
                mid = (left + right) / 2
                if step == "L":
                    b, right = lc, mid
                elif step == "R":
                    b, left = rc, mid
                else:
                    return False
            if tuple(map(str, b)) != leaf.coefficients or max(b) > 0:
                return False
            if strict and (min(b) == 0 or (left > 0 and b[0] == 0) or (right < 1 and b[-1] == 0)):
                return False
    return True
