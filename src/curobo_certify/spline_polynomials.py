"""Exact-rational spline extraction for the declared TimeOpt polynomial model.

The path is cuRobo's uniform cardinal cubic control polygon. Squared speed is
the clamped cubic B-spline with 32 controls and the nominal rational knots
0, 1/29, ..., 1. Float32 controls are interpreted as their exact binary values.
This extracts the mathematical polynomial model; parity with native FP32
kernel evaluation is a separate numerical gate.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np

from .boundary_reference import Span

Poly = tuple[Fraction, ...]  # ascending local powers


def _trim(p: Sequence[Fraction]) -> Poly:
    values = list(p) or [Fraction(0)]
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _add(a: Sequence[Fraction], b: Sequence[Fraction]) -> Poly:
    return _trim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
                  for i in range(max(len(a), len(b)))])


def _scale(p: Sequence[Fraction], c: Fraction) -> Poly:
    return _trim([c * x for x in p])


def _mul(a: Sequence[Fraction], b: Sequence[Fraction]) -> Poly:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return _trim(out)


def _affine_compose(poly: Sequence[Fraction], alpha: Fraction, beta: Fraction) -> Poly:
    """Return p(alpha + beta*u) exactly."""
    out: Poly = (Fraction(0),)
    linear = (alpha, beta)
    for coefficient in reversed(poly):
        out = _add(_mul(out, linear), (coefficient,))
    return out


def _exact_float(value) -> Fraction:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("spline controls must be finite")
    return Fraction.from_float(value)


def _timing_basis_on_span(knots: Sequence[Fraction], degree: int,
                          span_index: int) -> list[Poly]:
    """Cox-de Boor basis polynomials in local u on one nonempty knot span."""
    left, right = knots[degree + span_index], knots[degree + span_index + 1]
    if right <= left:
        raise ValueError("requested timing interval is not a nonempty knot span")
    h = right - left
    current: list[Poly] = []
    for i in range(len(knots) - 1):
        current.append((Fraction(1),) if knots[i] == left and knots[i + 1] == right
                       else (Fraction(0),))
    for p in range(1, degree + 1):
        next_level: list[Poly] = []
        for i in range(len(current) - 1):
            first_den = knots[i + p] - knots[i]
            second_den = knots[i + p + 1] - knots[i + 1]
            value: Poly = (Fraction(0),)
            if first_den:
                first = _mul((left - knots[i], h), current[i])
                value = _add(value, _scale(first, Fraction(1, 1) / first_den))
            if second_den:
                second = _mul((knots[i + p + 1] - left, -h), current[i + 1])
                value = _add(value, _scale(second, Fraction(1, 1) / second_den))
            next_level.append(value)
        current = next_level
    return current


def _path_span_polynomial(points: Sequence[Sequence[Fraction]], index: int) -> tuple[Poly, ...]:
    # Cubic uniform cardinal basis, rows are the four support basis functions;
    # columns are ascending powers of local u.
    rows = (
        (Fraction(1, 6), Fraction(-3, 6), Fraction(3, 6), Fraction(-1, 6)),
        (Fraction(4, 6), Fraction(0), Fraction(-6, 6), Fraction(3, 6)),
        (Fraction(1, 6), Fraction(3, 6), Fraction(3, 6), Fraction(-3, 6)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1, 6)),
    )
    return tuple(_trim([sum((points[index + k][joint] * rows[k][power]
                             for k in range(4)), Fraction(0))
                        for power in range(4)])
                 for joint in range(len(points[0])))


def extract_timeopt_spans(effective_control_points, timing_controls,
                          *, timing_degree: int = 3) -> tuple[Span, ...]:
    """Build exact local-power q/x on the union path and timing knot partition.

    ``effective_control_points`` must be the cloned complete cuRobo spline
    support, not the original action controls. ``timing_controls`` must contain
    exactly 32 cloned TimeOpt coefficients. Native FP32 parity is not asserted.
    """
    if hasattr(effective_control_points, "detach"):
        effective_control_points = effective_control_points.detach().cpu().numpy()
    if hasattr(timing_controls, "detach"):
        timing_controls = timing_controls.detach().cpu().numpy()
    points_array = np.asarray(effective_control_points)
    timing_array = np.asarray(timing_controls).reshape(-1)
    if points_array.ndim != 2 or points_array.shape[0] < 4 or points_array.shape[1] < 1:
        raise ValueError("effective spline control points must have shape [C,D], C>=4")
    if timing_array.shape != (32,) or timing_degree != 3:
        raise ValueError("TimeOpt extraction requires exactly 32 cubic timing controls")
    points = [[_exact_float(points_array[i, j]) for j in range(points_array.shape[1])]
              for i in range(points_array.shape[0])]
    timing = [_exact_float(value) for value in timing_array]

    path_count = points_array.shape[0] - 3
    path_breaks = [Fraction(i, path_count) for i in range(path_count + 1)]
    path_polys = [_path_span_polynomial(points, i) for i in range(path_count)]

    # The clamped cubic vector is [0,0,0] + breaks(0..1 by 1/29) + [1,1,1].
    timing_knots = ([Fraction(0)] * 3
                    + [Fraction(i, 29) for i in range(30)]
                    + [Fraction(1)] * 3)
    timing_count = len(timing_knots) - timing_degree - 1
    if timing_count != 32:
        raise AssertionError("internal timing knot topology does not yield 32 controls")
    timing_breaks = [timing_knots[timing_degree + i] for i in range(30)]
    timing_polys = []
    for i in range(29):
        basis = _timing_basis_on_span(timing_knots, timing_degree, i)
        timing_polys.append(_trim([sum((timing[k] * basis[k][power]
                                        for k in range(32) if power < len(basis[k])), Fraction(0))
                                   for power in range(4)]))

    cuts = sorted(set(path_breaks + timing_breaks))
    output = []
    for left, right in zip(cuts, cuts[1:]):
        mid = (left + right) / 2
        pi = min(int(mid * path_count), path_count - 1)
        pu0 = (left - path_breaks[pi]) * path_count
        pub = (right - left) * path_count
        q = tuple(_affine_compose(poly, pu0, pub) for poly in path_polys[pi])

        ti = min(int(mid * 29), 28)
        tleft = timing_breaks[ti]
        th = timing_breaks[ti + 1] - tleft
        tu0 = (left - tleft) / th
        tub = (right - left) / th
        x = _affine_compose(timing_polys[ti], tu0, tub)
        output.append(Span(left, right, q, x))
    return tuple(output)
