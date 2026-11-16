"""Rigorous forward error bounds for cuRobo's native FP32 TimeOpt sample kernel.

Scope
-----
This module bounds, for every float32 progress value ``s`` in [0, 1], the
difference between the values *reported* by the pinned native kernel
``sample_with_progress_kernel`` (cuRobo c301cb00, via ``sample_joint`` and
``timing_state``) and the exact mathematical model (exact cardinal cubic path of
the float32 effective controls and exact clamped cubic squared-speed spline of
the 32 float32 timing controls).

Since the native kernel computes every reported velocity/acceleration/jerk as a
deterministic function of the final progress ``s`` (and ``s`` is clamped to
[0,1] by ``progress_at_time``), a bound that holds for all ``s`` covers every
requested physical time, whatever the accuracy of the time->progress inversion.
It says nothing about whether ``s(t)`` is the exact inverse of t(s); that
separate *time-consistency* claim is not established here.

Arithmetic model (assumptions, all explicit and parameterised in ``ArithmeticModel``)
--------------------------------------------------------------------------------
The pinned kernels are compiled by NVRTC with ``--ftz=true --fmad=true
--prec-div=false --prec-sqrt=false`` (see the pinned kernel_config.py). We assume:

A1  float32 add/sub/mul and fused multiply-add are correctly rounded to nearest;
    results below 2**-126 may be flushed to zero (absolute error < 2**-126).
A2  FMA contraction may or may not be applied to any ``a*b+c``; every bound below
    charges a rounding for every product *and* every sum, which dominates both.
    No other reassociation is assumed (nvcc does not reassociate float math
    without fast-math).
A3  approximate division ``x/y`` has error at most ``div_ulps`` units in the last
    place (NVIDIA documents 2 ulp for -prec-div=false).
A4  ``sqrtf`` has error at most ``sqrt_ulps`` ulp (NVIDIA documents 1 ulp for
    -prec-sqrt=false on compute capability >= 5.2; default here is 3).
A5  the compiled binary implements the C++ source semantics otherwise, inputs
    are the captured float32 arrays and contain no subnormal numbers.

These are vendor specifications and source-level assumptions, not proofs about
the silicon. Every result produced from this module carries them.

Bound mechanics: each quantity is an ``ET`` = (outward interval enclosing the
exact model value over a parameter box, upper bound on |native - exact|). All
bound arithmetic is done in binary64 with explicit upward rounding
(``math.nextafter``) so the bounds themselves are not under-estimated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

INF = math.inf
U32 = 2.0 ** -24          # float32 unit roundoff (round to nearest)
ULP_REL = 2.0 ** -23      # 1 ulp relative to the value, upper bound


def up(x: float) -> float:
    return math.nextafter(x, INF)


def dn(x: float) -> float:
    return math.nextafter(x, -INF)


def fup(value: Fraction) -> float:
    """Smallest-effort upward float bound of an exact rational."""
    return up(float(value))


def fdn(value: Fraction) -> float:
    return dn(float(value))


@dataclass(frozen=True)
class ArithmeticModel:
    unit: float = U32
    div_ulps: float = 2.0
    sqrt_ulps: float = 3.0
    ftz_abs: float = 2.0 ** -126

    @property
    def div_rel(self) -> float:
        return up(self.div_ulps * ULP_REL)

    @property
    def sqrt_rel(self) -> float:
        return up(self.sqrt_ulps * ULP_REL)

    def describe(self) -> dict:
        return {"unit_roundoff": self.unit, "div_ulps": self.div_ulps, "sqrt_ulps": self.sqrt_ulps,
                "ftz_abs": self.ftz_abs,
                "assumptions": ["A1 RN add/sub/mul/fma, FTZ absolute 2^-126",
                                "A2 FMA contraction optional (bounds dominate both)",
                                f"A3 approximate division <= {self.div_ulps} ulp",
                                f"A4 sqrtf <= {self.sqrt_ulps} ulp",
                                "A5 binary implements source semantics; no subnormal inputs"]}


@dataclass(frozen=True)
class ET:
    """Exact value in [lo, hi]; |native - exact| <= e."""
    lo: float
    hi: float
    e: float

    @property
    def mag(self) -> float:
        return max(abs(self.lo), abs(self.hi))

    @staticmethod
    def exact(value: Fraction | float | int) -> "ET":
        f = Fraction(value)
        return ET(fdn(f), fup(f), 0.0)

    @staticmethod
    def interval(lo: Fraction, hi: Fraction) -> "ET":
        return ET(fdn(Fraction(lo)), fup(Fraction(hi)), 0.0)


def _round(m: ArithmeticModel, mag: float, prop: float, rel: float) -> float:
    return up(prop + up(rel * up(mag + prop)) + m.ftz_abs)


def et_add(m: ArithmeticModel, a: ET, b: ET) -> ET:
    lo, hi = dn(a.lo + b.lo), up(a.hi + b.hi)
    prop = up(a.e + b.e)
    return ET(lo, hi, _round(m, max(abs(lo), abs(hi)), prop, m.unit))


def et_neg(a: ET) -> ET:
    return ET(-a.hi, -a.lo, a.e)


def et_sub(m: ArithmeticModel, a: ET, b: ET) -> ET:
    return et_add(m, a, et_neg(b))


def _iv_mul(a: ET, b: ET) -> tuple[float, float]:
    products = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    return dn(min(products)), up(max(products))


def et_mul(m: ArithmeticModel, a: ET, b: ET) -> ET:
    lo, hi = _iv_mul(a, b)
    prop = up(up(up(a.mag * b.e) + up(b.mag * a.e)) + up(a.e * b.e))
    return ET(lo, hi, _round(m, max(abs(lo), abs(hi)), prop, m.unit))


def et_div(m: ArithmeticModel, a: ET, b: ET) -> ET:
    if b.lo <= 0 <= b.hi:
        raise ValueError("divisor range contains zero")
    bmin = min(abs(b.lo), abs(b.hi))
    if bmin <= b.e:
        raise ValueError("divisor error can reach zero")
    quotients = [Fraction(x) / Fraction(y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
    lo, hi = fdn(min(quotients)), fup(max(quotients))
    mag = max(abs(lo), abs(hi))
    prop = up(up(a.e + up(mag * b.e)) / dn(bmin - b.e))
    return ET(lo, hi, _round(m, mag, prop, m.div_rel))


def et_sqrt_nonneg(m: ArithmeticModel, a: ET) -> ET:
    """native sqrtf(fmaxf(0, a~)) versus exact sqrt(a), exact a >= 0 required."""
    if a.lo < 0:
        # The exact quantity is nonnegative (positivity is a certified obligation);
        # clip the overestimated enclosure.
        a = ET(0.0, max(a.hi, 0.0), a.e)
    lo, hi = dn(math.sqrt(max(a.lo, 0.0))), up(math.sqrt(max(a.hi, 0.0)))
    # fmaxf(0,.) is 1-Lipschitz towards a nonnegative target; then
    # |sqrt(y)-sqrt(x)| <= min(sqrt|y-x|, |y-x|/sqrt(xmin)).
    prop = up(math.sqrt(a.e))
    if a.lo > 0:
        prop = min(prop, up(a.e / dn(math.sqrt(a.lo))))
    return ET(max(lo, 0.0), hi, _round(m, hi, prop, m.sqrt_rel))


def const(value) -> ET:
    return ET.exact(value)


# ---------------------------------------------------------------------------
# Native geometry: sample_joint -> BSplineContext<3> with BSplineBasisPolynomial<3>
# ---------------------------------------------------------------------------

def _native_velocity_basis(m, t: ET) -> list[ET]:
    one, three, six, nine, twelve = (const(v) for v in (1, 3, 6, 9, 12))
    t2 = et_mul(m, t, t)
    omt = et_sub(m, one, t)
    b0 = et_div(m, et_mul(m, et_mul(m, const(-3), omt), omt), six)
    b1 = et_div(m, et_sub(m, et_mul(m, nine, t2), et_mul(m, twelve, t)), six)
    b2 = et_div(m, et_add(m, et_add(m, et_mul(m, const(-9), t2), et_mul(m, six, t)), three), six)
    b3 = et_div(m, et_mul(m, three, t2), six)
    return [b0, b1, b2, b3]


def _native_acceleration_basis(m, t: ET) -> list[ET]:
    six = const(6)
    b0 = et_div(m, et_mul(m, six, et_sub(m, const(1), t)), six)
    b1 = et_div(m, et_sub(m, et_mul(m, const(18), t), const(12)), six)
    b2 = et_div(m, et_add(m, et_mul(m, const(-18), t), six), six)
    b3 = et_div(m, et_mul(m, six, t), six)
    return [b0, b1, b2, b3]


def _native_jerk_basis() -> list[ET]:
    return [const(-1), const(3), const(-3), const(1)]


def _dot(m, values: Sequence[ET], basis: Sequence[ET]) -> ET:
    result = const(0)
    for v, b in zip(values, basis):
        result = et_add(m, result, et_mul(m, v, b))
    return result


def native_geometry(m: ArithmeticModel, support: Sequence[float], spans: int,
                    t_lo: Fraction, t_hi: Fraction) -> tuple[ET, ET, ET]:
    """Q1,Q2,Q3 = d^k q/ds^k (global s) on one segment, local t in [t_lo,t_hi].

    ``support`` are the four exact float32 control values. The native evaluation
    point t is the exact float32 ``coordinate - segment`` (Sterbenz exact), so the
    exact reference is the segment polynomial at that very point.
    """
    s = [Fraction(float(v)) for v in support]
    centered = [const(0)] + [ET(fdn(s[k] - s[0]), fup(s[k] - s[0]),
                                _round(m, abs(float(s[k] - s[0])), 0.0, m.unit))
                             for k in range(1, 4)]
    t = ET.interval(t_lo, t_hi)
    kdt = et_div(m, const(1), const(spans))
    kdt2 = et_mul(m, kdt, kdt)
    kdt3 = et_mul(m, kdt2, kdt)
    q1 = et_div(m, _dot(m, centered, _native_velocity_basis(m, t)), kdt)
    q2 = et_div(m, _dot(m, centered, _native_acceleration_basis(m, t)), kdt2)
    q3 = et_div(m, _dot(m, centered, _native_jerk_basis()), kdt3)
    return q1, q2, q3


# ---------------------------------------------------------------------------
# Native timing: timing_state (FP32 table polynomial, Horner in local)
# ---------------------------------------------------------------------------

def exact_timing_table() -> list[list[list[Fraction]]]:
    """Exact T[m][P][k]: coefficient of (s-m/29)^(3-P) for control m+k, k=0..3."""
    from .spline_polynomials import _timing_basis_on_span
    knots = [Fraction(0)] * 3 + [Fraction(i, 29) for i in range(30)] + [Fraction(1)] * 3
    table = []
    for span in range(29):
        basis = _timing_basis_on_span(knots, 3, span)  # 32 basis polys in local u
        rows = []
        for P in range(4):
            p = 3 - P
            row = []
            for k in range(4):
                poly = basis[span + k]
                c = poly[p] if p < len(poly) else Fraction(0)
                row.append(c * Fraction(29) ** p)
            rows.append(row)
        table.append(rows)
    return table


def native_timing(m: ArithmeticModel, native_table, exact_table, controls: Sequence[float],
                  span: int, local_lo: Fraction, local_hi: Fraction) -> tuple[ET, ET, ET]:
    """X0,X1,X2 of exact timing polynomial ``span`` at the native local point.

    ``native_table[span][P][span+k]`` is the float32 table entry for control
    span+k (layout [29][4][32]); its exact difference from ``exact_table`` is
    charged as input error.
    """
    a = []
    for P in range(4):
        acc = const(0)
        for k in range(4):
            exact = exact_table[span][P][k]
            nat = Fraction(float(native_table[span][P][span + k]))  # layout [span][power][control]
            entry = ET(fdn(exact), fup(exact), fup(abs(nat - exact)))
            c = Fraction(float(controls[span + k]))
            acc = et_add(m, acc, et_mul(m, entry, ET(fdn(c), fup(c), 0.0)))
        a.append(acc)
    loc = ET.interval(local_lo, local_hi)
    x0 = et_add(m, et_mul(m, et_add(m, et_mul(m, et_add(m, et_mul(m, a[0], loc), a[1]), loc), a[2]), loc), a[3])
    x1 = et_add(m, et_mul(m, et_add(m, et_mul(m, et_mul(m, const(3), a[0]), loc), et_mul(m, const(2), a[1])), loc), a[2])
    x2 = et_add(m, et_mul(m, et_mul(m, const(6), a[0]), loc), et_mul(m, const(2), a[1]))
    return x0, x1, x2


def widen(a: ET, extra: float) -> ET:
    return ET(dn(a.lo - extra), up(a.hi + extra), up(a.e + extra))


def native_outputs(m: ArithmeticModel, q: tuple[ET, ET, ET], x: tuple[ET, ET, ET]) -> dict:
    """values[1..3] of sample_with_progress_kernel for one joint."""
    q1, q2, q3 = q
    x0, x1, x2 = x
    speed = et_sqrt_nonneg(m, x0)
    vel = et_mul(m, q1, speed)
    acc = et_add(m, et_mul(m, q2, x0), et_mul(m, et_mul(m, const(0.5), q1), x1))
    inner = et_add(m, et_add(m, et_mul(m, q3, x0), et_mul(m, et_mul(m, const(1.5), q2), x1)),
                   et_mul(m, et_mul(m, const(0.5), q1), x2))
    jerk = et_mul(m, speed, inner)
    return {"velocity": vel, "acceleration": acc, "jerk": jerk, "speed": speed}


def _poly_deriv_bound(coeffs_desc: Sequence[Fraction], order: int, lo: Fraction, hi: Fraction) -> Fraction:
    """Upper bound of |d^order p| on [lo,hi], p in powers of local (descending P order)."""
    asc = [coeffs_desc[3 - p] for p in range(4)]  # ascending powers
    for _ in range(order):
        asc = [k * asc[k] for k in range(1, len(asc))] or [Fraction(0)]
    r = max(abs(lo), abs(hi))
    return sum((abs(c) * r ** i for i, c in enumerate(asc)), Fraction(0))


@dataclass(frozen=True)
class SpanBound:
    span_index: int
    joint: int
    velocity: float
    acceleration: float
    jerk: float
    pieces: int


def common_span_bounds(m: ArithmeticModel, spans_model, geometry, timing_controls, native_table,
                       *, pieces: int = 16, delta: Fraction = Fraction(1, 10 ** 6),
                       exact_table=None) -> list[list[SpanBound]]:
    """Per common span and joint: native-vs-model error bounds E_v, E_a, E_j.

    ``spans_model`` are the exact common spans (left/right as Fractions).
    ``delta`` (in s units) must dominate every native progress offset:
      * timing point s_t = m/29 + fl(s - fl(m/29)):        |s_t - s| <= 5.04u
      * geometry point (k+t)/S, t = coordinate-k or the FMA-contracted
        RN(s*S-k) (possibly slightly outside [0,1]):        |.. - s| <= u
      * s itself may exceed 1 by <= 4u (approximate (cell+1)/intervals in
        progress_at_time); clamping the geometry reference: <= 5u more
    so |s_t - s_g| <= 11.04u ~ 6.6e-7 < delta = 1e-6 (16.7u).
    The geometry polynomial is therefore evaluated for t in [t_lo-delta*S,
    t_hi+delta*S]; the reference is the same segment polynomial at the clamped
    point, which lies in the certified closed common span, plus a Lipschitz term.
    """
    if delta < Fraction(12) * Fraction(2) ** -24:
        raise ValueError("delta must dominate the 11.04u native progress offset")
    exact_table = exact_table or exact_timing_table()
    S = len(geometry) - 3
    dof = len(geometry[0])
    # Exact timing span polynomials (descending P order coefficients) for shift bounds.
    tpoly = []
    for span in range(29):
        tpoly.append([sum((exact_table[span][P][k] * Fraction(float(timing_controls[span + k]))
                           for k in range(4)), Fraction(0)) for P in range(4)])
    out = []
    for ci, sp in enumerate(spans_model):
        L, R = Fraction(sp.left), Fraction(sp.right)
        k = min(int(((L + R) / 2) * S), S - 1)
        mid_t = min(int(((L + R) / 2) * 29), 28)
        rows = []
        cuts = [L + (R - L) * Fraction(i, pieces) for i in range(pieces + 1)]
        worst = [[0.0, 0.0, 0.0] for _ in range(dof)]
        for a, b in zip(cuts, cuts[1:]):
            t_lo, t_hi = a * S - k, b * S - k
            for mspan in (mid_t - 1, mid_t, mid_t + 1):
                if not 0 <= mspan <= 28:
                    continue
                left_m = Fraction(mspan, 29)
                lo_s = max(a - delta, left_m - delta)
                hi_s = min(b + delta, left_m + Fraction(1, 29) + delta)
                if lo_s > hi_s:
                    continue
                x = native_timing(m, native_table, exact_table, timing_controls, mspan,
                                  lo_s - left_m, hi_s - left_m)
                # Shift: exact X_m at s_t versus exact spline at s_g, |s_t-s_g|<=delta.
                ext_lo, ext_hi = lo_s - left_m - delta, hi_s - left_m + delta
                shifts = []
                for i in range(3):
                    lip = _poly_deriv_bound(tpoly[mspan], i + 1, ext_lo, ext_hi)
                    shift = lip * delta
                    if mspan != mid_t or a - delta < left_m or b + delta > left_m + Fraction(1, 29):
                        # possible cross-knot pairing: C2 spline, jump in X'''.
                        jumps = [abs(6 * tpoly[n][0] - 6 * tpoly[mspan][0])
                                 for n in (mspan - 1, mspan + 1) if 0 <= n <= 28]
                        jump = max(jumps) if jumps else Fraction(0)
                        shift += jump * delta ** (3 - i) / math.factorial(3 - i)
                    shifts.append(fup(shift))
                x = tuple(widen(xi, s) for xi, s in zip(x, shifts))
                for j in range(dof):
                    support = [geometry[k + r][j] for r in range(4)]
                    q = native_geometry(m, support, S, t_lo - delta * S, t_hi + delta * S)
                    # Same-polynomial Lipschitz shift to the clamped reference point:
                    # |Q_i(t) - Q_i(clamp t)| <= sup|Q_{i+1}| * delta (global s units);
                    # Q3 is constant on a segment, so its shift is zero.
                    q = (widen(q[0], fup(Fraction(q[1].mag) * delta)),
                         widen(q[1], fup(Fraction(q[2].mag) * delta)), q[2])
                    o = native_outputs(m, q, x)
                    worst[j][0] = max(worst[j][0], o["velocity"].e)
                    worst[j][1] = max(worst[j][1], o["acceleration"].e)
                    worst[j][2] = max(worst[j][2], o["jerk"].e)
        out.append([SpanBound(ci, j, *worst[j], pieces) for j in range(dof)])
    return out


def tightened_limits(limits: dict, bounds_row: Sequence[SpanBound]) -> tuple[list, list, list]:
    """Exact rational limits reduced by the per-joint native error bounds."""
    v, a, jl = [], [], []
    for j, b in enumerate(bounds_row):
        for name, err, dest in (("velocity", b.velocity, v), ("acceleration", b.acceleration, a),
                                ("jerk", b.jerk, jl)):
            lim = Fraction(float(limits[name][j])) - Fraction(err)
            if lim <= 0:
                raise ValueError(f"native error bound exceeds {name} limit for joint {j}")
            dest.append(lim)
    return v, a, jl


# ---------------------------------------------------------------------------
# Native-reported-value K3 certificate: exact model against tightened limits.
# ---------------------------------------------------------------------------

NATIVE_SCOPE = "NATIVE_REPORTED_K3_CONDITIONAL_ON_DOCUMENTED_FP32_SEMANTICS"


def certify_native_reported(spans, targets, limits: dict, bounds, *, max_depth: int = 12,
                            max_nodes: int = 65536):
    """Exact Bernstein certificate of every common span against limits reduced by
    the rigorous native evaluation bounds of that span.

    Returns (status, detail, nodes, leaves). Status CERTIFIED means: under the
    ArithmeticModel assumptions, for every float32 progress s in [0,1] the native
    kernel's reported velocity, acceleration and jerk satisfy the original
    physical limits. It is NOT a time-consistency (s(t) inversion) certificate.
    """
    from .boundary_reference import check_boundaries
    from .exact_k3_reference import _check_one, bernstein_coeffs, build_residuals
    boundary = check_boundaries(spans, targets)
    if boundary.status != "PASS":
        return "UNKNOWN_BOUNDARY" if boundary.status != "VIOLATED" else "VIOLATED", \
            "boundary/regularity: " + boundary.status, 0, ()
    nodes, leaves = 0, []
    for piece, span in enumerate(spans):
        try:
            v, a, j = tightened_limits(limits, bounds[piece])
        except ValueError as exc:
            return "UNKNOWN_NATIVE_BOUND", f"span {piece}: {exc}", nodes, tuple(leaves)
        for family, joint, power in build_residuals(span, v, a, j):
            rid = f"span{piece}/{family}/joint{joint}"
            state, got, nodes, detail = _check_one(bernstein_coeffs(power), rid, family == "positive_x",
                                                   max_depth, max_nodes, nodes)
            leaves.extend(got)
            if state != "CERTIFIED":
                # A positive exact witness against TIGHTENED limits is not a native
                # violation: the native value may still be within the original limit.
                if state == "VIOLATED":
                    state = "UNKNOWN_TIGHTENED_WITNESS"
                return state, detail, nodes, tuple(leaves)
    return "CERTIFIED", "all spans certified against native-error-tightened limits", nodes, tuple(leaves)
