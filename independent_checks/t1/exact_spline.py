"""Independent exact model of the TimeOpt squared-speed spline x(s).

Written for the R0 review WITHOUT using curobo_certify.spline_polynomials,
native_error_bound.exact_timing_table or time_law_grid. Method:

* float32 -> exact rational by decoding the IEEE-754 bit pattern by hand
  (sign, biased exponent, 23-bit fraction), not by Fraction(float(x));
* the clamped cubic B-spline of the 32 controls on the knot vector
  [0,0,0,0, 1/29, ..., 28/29, 1,1,1,1]  (upstream CubicTimingBasis uses
  np.r_[zeros(3), linspace(0,1,30), ones(3)], i.e. the same vector with the
  interior breaks i/29; the EXACT model takes them as exact rationals);
* per span m (s in [m/29, (m+1)/29]) the cubic is recovered by evaluating
  x exactly with de Boor's algorithm at four distinct rational points of
  the span and solving the 4x4 Vandermonde system exactly (Newton divided
  differences).  Result: ascending power coefficients in w = s - m/29.
"""
from __future__ import annotations

import struct
from fractions import Fraction as Q

SPANS = 29
NCTRL = 32
DEGREE = 3


def f32_to_rational(value) -> Q:
    """Exact rational value of a float32 (given as float/np.float32), by bit decoding."""
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    # struct.pack('<f') rounds a binary64 input to float32; callers pass float32 values,
    # so check the round-trip is exact.
    if float(struct.unpack("<f", struct.pack("<I", bits))[0]) != float(value):
        raise ValueError("value is not exactly representable as float32")
    return f32_bits_to_rational(bits)


def f32_bits_to_rational(bits: int) -> Q:
    sign = -1 if bits >> 31 else 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        raise ValueError("Inf/NaN has no rational value")
    if exp == 0:  # zero or subnormal: frac * 2^-149
        return sign * Q(frac, 1 << 149)
    mant = (1 << 23) | frac
    e = exp - 127 - 23
    return sign * (Q(mant * (1 << e)) if e >= 0 else Q(mant, 1 << -e))


def knot_vector() -> list[Q]:
    return [Q(0)] * 4 + [Q(i, SPANS) for i in range(1, SPANS)] + [Q(1)] * 4


def de_boor(ctrl: list[Q], s: Q, span: int) -> Q:
    """Exact value of the clamped cubic spline at s, using knot interval `span` (0..28).

    The interval index in the 36-knot vector is i = span + 3, with
    U[i] = span/29 <= s <= U[i+1] = (span+1)/29 (closed: we evaluate the span's
    polynomial, so endpoints use this span's piece)."""
    U = knot_vector()
    i = span + DEGREE
    if not (U[i] <= s <= U[i + 1]):
        raise ValueError("s outside requested span")
    d = [ctrl[j + i - DEGREE] for j in range(DEGREE + 1)]
    for r in range(1, DEGREE + 1):
        for j in range(DEGREE, r - 1, -1):
            left = U[j + i - DEGREE]
            right = U[j + 1 + i - r]
            alpha = (s - left) / (right - left)
            d[j] = (1 - alpha) * d[j - 1] + alpha * d[j]
    return d[DEGREE]


def span_cubic(ctrl: list[Q], span: int) -> tuple[Q, Q, Q, Q]:
    """Ascending coefficients (c0..c3) of x(s) = sum c_i w^i, w = s - span/29, on the span."""
    a = Q(span, SPANS)
    h = Q(1, SPANS)
    ws = [h * k / 3 for k in range(4)]  # 0, h/3, 2h/3, h
    ys = [de_boor(ctrl, a + w, span) for w in ws]
    # Newton divided differences, then expand to monomial basis.
    n = 4
    coef = list(ys)
    for lvl in range(1, n):
        for i in range(n - 1, lvl - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (ws[i] - ws[i - lvl])
    poly = [Q(0)] * n
    # Horner on Newton form: p = c0 + (w-w0)(c1 + (w-w1)(c2 + (w-w2)c3))
    acc = [coef[n - 1]]
    for i in range(n - 2, -1, -1):
        # acc * (w - ws[i]) + coef[i]
        new = [Q(0)] * (len(acc) + 1)
        for k, c in enumerate(acc):
            new[k + 1] += c
            new[k] -= c * ws[i]
        new[0] += coef[i]
        acc = new
    for k, c in enumerate(acc):
        poly[k] = c
    return tuple(poly)


def model_from_controls(timing_controls) -> list[tuple[Q, Q, Q, Q]]:
    """29 exact span cubics from 32 float32 timing controls."""
    vals = list(timing_controls)
    if len(vals) != NCTRL:
        raise ValueError("expected 32 timing controls")
    ctrl = [f32_to_rational(v) for v in vals]
    return [span_cubic(ctrl, m) for m in range(SPANS)]


def model_from_rationals(ctrl: list[Q]) -> list[tuple[Q, Q, Q, Q]]:
    if len(ctrl) != NCTRL:
        raise ValueError("expected 32 controls")
    return [span_cubic([Q(c) for c in ctrl], m) for m in range(SPANS)]


def eval_cubic(c, w: Q) -> Q:
    return ((c[3] * w + c[2]) * w + c[1]) * w + c[0]


def deriv(c):
    return tuple(Q(i) * c[i] for i in range(1, len(c))) or (Q(0),)


def continuity_defects(model) -> list[tuple[int, int, Q]]:
    """Exact jumps of x, x', x'' at interior knots (must be zero for C2)."""
    out = []
    h = Q(1, SPANS)
    for m in range(SPANS - 1):
        L, R = model[m], model[m + 1]
        dl, dr = list(L), list(R)
        for order in range(3):
            jump = eval_cubic_any(dr, Q(0)) - eval_cubic_any(dl, h)
            if jump != 0:
                out.append((m + 1, order, jump))
            dl, dr = list(deriv(dl)), list(deriv(dr))
    return out


def eval_cubic_any(c, w: Q) -> Q:
    acc = Q(0)
    for coef in reversed(list(c)):
        acc = acc * w + coef
    return acc
