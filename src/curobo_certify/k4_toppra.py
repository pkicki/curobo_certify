"""Exact K4 spans for a TOPP-RA time parametrization of a cuRobo geometric path.

TOPP-RA (Pham & Pham 2018; toppra package) returns the squared path speed x_i = sd_i^2 at gridpoints s_i with a
constant path acceleration on each grid interval, i.e. x(s) is piecewise LINEAR in s. That trajectory is
represented exactly (from the stored float64 gridpoints and sd values) on the common refinement of the cuRobo
geometry span partition and the TOPP-RA grid: q is the exact cuRobo polynomial re-expanded on each sub-interval,
x is the exact linear interpolant. x' jumps at gridpoints (torque is discontinuous there); every closed sub-span is
certified with its own one-sided polynomials, so both one-sided torques are covered.
"""
from __future__ import annotations

from fractions import Fraction as Q
from math import comb

from .boundary_reference import Span


def compose_sub(p, ua: Q, ub: Q):
    """r(v) = p(ua + (ub - ua) v), exact."""
    d = ub - ua
    out = [Q(0)] * len(p)
    for k, c in enumerate(p):
        # c (ua + d v)^k = c sum_j C(k,j) ua^(k-j) d^j v^j
        for j in range(k + 1):
            out[j] += c * comb(k, j) * ua ** (k - j) * d ** j
    return tuple(out)


def toppra_spans(spans, grid, x_grid):
    """spans: cuRobo Span tuple on [0,1]; grid: increasing Fractions with grid[0]=0, grid[-1]=1; x_grid: Fractions >= 0."""
    if grid[0] != 0 or grid[-1] != 1 or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("invalid TOPP-RA grid")
    if len(x_grid) != len(grid) or any(x < 0 for x in x_grid):
        raise ValueError("invalid x grid")
    cuts = sorted(set([sp.left for sp in spans] + [spans[-1].right] + list(grid)))
    out = []
    k = g = 0
    for a, b in zip(cuts, cuts[1:]):
        while spans[k].right <= a:
            k += 1
        while grid[g + 1] <= a:
            g += 1
        sp = spans[k]
        h = sp.right - sp.left
        ua, ub = (a - sp.left) / h, (b - sp.left) / h
        q = tuple(compose_sub(p, ua, ub) for p in sp.q)
        s0, s1, x0, x1 = grid[g], grid[g + 1], x_grid[g], x_grid[g + 1]
        slope = (x1 - x0) / (s1 - s0)
        xa = x0 + slope * (a - s0)
        x = (xa, slope * (b - a))
        out.append(Span(a, b, q, x))
    return tuple(out)


def toppra_duration(grid, sd):
    """Exact duration of the constant-acceleration parametrization: sum 2 ds / (sd_i + sd_{i+1})."""
    t = Q(0)
    for a, b, v0, v1 in zip(grid, grid[1:], sd, sd[1:]):
        if v0 + v1 <= 0:
            return None
        t += 2 * (b - a) / (v0 + v1)
    return t


def moving_interval(spans):
    """[s_a, s_b]: union of spans where some joint polynomial is non-constant (cuRobo pads the spline with stationary
    start/goal spans; a TOPP-RA user would parametrize only the moving part)."""
    mv = [i for i, sp in enumerate(spans) if any(any(c != 0 for c in p[1:]) for p in sp.q)]
    if not mv:
        raise ValueError("path never moves")
    return spans[mv[0]].left, spans[mv[-1]].right, mv[0], mv[-1]


def trimmed(spans):
    """Spans of the moving part only, re-parametrized to sigma = (s - s_a)/(s_b - s_a) in [0, 1].
    Local q polynomials are unchanged (local variable u); x is not used by TOPP-RA (only geometry)."""
    sa, sb, i0, i1 = moving_interval(spans)
    L = sb - sa
    return tuple(Span((sp.left - sa) / L, (sp.right - sa) / L, sp.q, tuple(v / (L * L) for v in sp.x))
                 for sp in spans[i0:i1 + 1]), (sa, sb)
