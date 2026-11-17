"""Vectorized outward-rounded binary64 interval arithmetic for K4 (Panda interval RNEA).

Every operation returns [lo, hi] arrays that enclose the exact real result for every
real input inside the operand intervals, assuming IEEE-754 binary64 round-to-nearest
arithmetic (each basic result is within half an ulp, then widened by one ulp with
np.nextafter). sin/cos use a Maclaurin polynomial evaluated in this interval arithmetic
at the box midpoint, an exact-rational Lagrange remainder bound and the 1-Lipschitz
property for the box radius; |midpoint| > 8 falls back to [-1, 1]. Non-finite results
fail closed (callers must check `finite()`).

Scope: CPU reference/production for K4 research; not FTZ/DAZ-safe if the process enables
them (checked at import), not valid under non-default rounding modes.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial

import numpy as np

INF = np.inf


def _dn(x):
    return np.nextafter(x, -INF)


def _up(x):
    return np.nextafter(x, INF)


def _check_fp_environment():
    # subnormal arithmetic must not be flushed (FTZ/DAZ would break nextafter enclosures)
    tiny = np.float64(5e-324)
    if not (tiny > 0 and tiny / 2 == 0 and np.float64(2.2250738585072014e-308) / 2 > 0):
        raise RuntimeError("floating-point environment flushes subnormals; K4 intervals unsound")
    if np.float64(1.0) + np.float64(2.0 ** -53) != 1.0 or np.float64(1.0) + np.float64(2.0 ** -52) == 1.0:
        raise RuntimeError("binary64 round-to-nearest not detected")


_check_fp_environment()


class IA:
    """Array of closed intervals [lo, hi] (numpy float64 arrays of identical shape)."""

    __slots__ = ("lo", "hi")
    __array_priority__ = 1000

    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, dtype=np.float64)
        hi = lo if hi is None else np.asarray(hi, dtype=np.float64)
        self.lo, self.hi = np.broadcast_arrays(lo, hi)
        self.lo = np.array(self.lo, dtype=np.float64)
        self.hi = np.array(self.hi, dtype=np.float64)

    # ---- constructors -------------------------------------------------------------
    @staticmethod
    def const(x, shape=()):
        """Enclosure of a real constant given as float (exact) or Fraction (outward)."""
        if isinstance(x, Fraction):
            f = float(x)
            lo, hi = (f, f) if Fraction(f) == x else (float(_dn(f)), float(_up(f)))
            return IA(np.full(shape, lo), np.full(shape, hi))
        return IA(np.full(shape, float(x)))

    @staticmethod
    def of(x, like=None):
        if isinstance(x, IA):
            return x
        if isinstance(x, Fraction):
            return IA.const(x, () if like is None else like.lo.shape)
        return IA(np.asarray(x, dtype=np.float64))

    # ---- arithmetic -----------------------------------------------------------------
    def __add__(self, o):
        if getattr(o, '_is_dual', False):
            return NotImplemented
        o = IA.of(o)
        return IA(_dn(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return IA(-self.hi, -self.lo)

    def __sub__(self, o):
        if getattr(o, '_is_dual', False):
            return NotImplemented
        o = IA.of(o)
        return IA(_dn(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, o):
        if getattr(o, '_is_dual', False):
            return NotImplemented
        return IA.of(o) - self

    def __mul__(self, o):
        if getattr(o, '_is_dual', False):
            return NotImplemented
        o = IA.of(o)
        p = np.stack(np.broadcast_arrays(self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi))
        # 0 * inf -> nan would hide a blow-up: map nan to +-inf so finite() fails
        lo = np.where(np.isnan(p).any(0), -INF, p.min(0))
        hi = np.where(np.isnan(p).any(0), INF, p.max(0))
        return IA(_dn(lo), _up(hi))

    __rmul__ = __mul__

    def scale(self, c: float):
        """Multiply by an exactly representable float constant."""
        c = float(c)
        a, b = self.lo * c, self.hi * c
        return IA(_dn(np.minimum(a, b)), _up(np.maximum(a, b)))

    def sqr(self):
        a, b = self.lo * self.lo, self.hi * self.hi
        lo = np.where((self.lo <= 0) & (self.hi >= 0), 0.0, _dn(np.minimum(a, b)))
        return IA(lo, _up(np.maximum(a, b)))

    # ---- utilities ------------------------------------------------------------------
    def finite(self):
        return bool(np.isfinite(self.lo).all() and np.isfinite(self.hi).all() and (self.lo <= self.hi).all())

    def mid(self):
        return 0.5 * self.lo + 0.5 * self.hi

    def rad_up(self):
        m = self.mid()
        return _up(np.maximum(_up(self.hi - m), _up(m - self.lo)))

    def hull(self, o):
        return IA(np.minimum(self.lo, o.lo), np.maximum(self.hi, o.hi))

    def __getitem__(self, k):
        return IA(self.lo[k], self.hi[k])

    @property
    def shape(self):
        return self.lo.shape

    def __repr__(self):
        return f"IA({self.lo!r}, {self.hi!r})"


def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


# ---- rigorous sin / cos ----------------------------------------------------------------
_ORDER = 30  # sine degree 61, cosine degree 60
_MAXMID = 8.0


def _taylor_point(m: np.ndarray, cosine: bool) -> IA:
    """Interval enclosure of the Maclaurin polynomial at point m (Horner in x^2)."""
    x = IA(m)
    x2 = x.sqr()
    # coefficients (-1)^k / (2k + e)! as outward intervals
    e = 0 if cosine else 1
    acc = IA.const(Fraction((-1) ** _ORDER, factorial(2 * _ORDER + e)), m.shape)
    for k in range(_ORDER - 1, -1, -1):
        acc = acc * x2 + IA.const(Fraction((-1) ** k, factorial(2 * k + e)), m.shape)
    return acc if cosine else acc * x


def _remainder_bound(xmax: float, cosine: bool) -> float:
    d = 2 * _ORDER + (0 if cosine else 1)
    r = Fraction(xmax) ** (d + 1) / factorial(d + 1)
    f = float(r)
    return f if Fraction(f) >= r else float(_up(f))


def _trig(a: IA, cosine: bool) -> IA:
    m = a.mid()
    rad = a.rad_up()
    big = ~(np.abs(m) <= _MAXMID)  # includes nan
    ms = np.where(big, 0.0, m)
    t = _taylor_point(ms, cosine)
    rem = _remainder_bound(float(np.max(np.abs(ms))) if ms.size else 0.0, cosine)
    lo = _dn(_dn(t.lo - rem) - rad)
    hi = _up(_up(t.hi + rem) + rad)
    lo = np.maximum(lo, -1.0)
    hi = np.minimum(hi, 1.0)
    lo = np.where(big, -1.0, lo)
    hi = np.where(big, 1.0, hi)
    return IA(lo, hi)


def isin(a):
    if getattr(a, "_is_dual", False):
        return a.sin()
    return _trig(a, False)


def icos(a):
    if getattr(a, "_is_dual", False):
        return a.cos()
    return _trig(a, True)


class DIA:
    """Forward-mode dual interval: value v and derivative d (w.r.t. one scalar parameter), both IA.
    Every operation encloses value and derivative of the exact function over the box (chain rule
    evaluated in outward interval arithmetic). Used for rigorous first-order Taylor models."""

    __slots__ = ("v", "d")
    _is_dual = True
    __array_priority__ = 2000

    def __init__(self, v, d):
        self.v, self.d = v, d

    @staticmethod
    def lift(o):
        if isinstance(o, DIA):
            return o
        o = IA.of(o)
        return DIA(o, IA(np.zeros_like(o.lo)))

    def __add__(self, o):
        o = DIA.lift(o)
        return DIA(self.v + o.v, self.d + o.d)

    __radd__ = __add__

    def __neg__(self):
        return DIA(-self.v, -self.d)

    def __sub__(self, o):
        o = DIA.lift(o)
        return DIA(self.v - o.v, self.d - o.d)

    def __rsub__(self, o):
        return DIA.lift(o) - self

    def __mul__(self, o):
        o = DIA.lift(o)
        return DIA(self.v * o.v, self.d * o.v + self.v * o.d)

    __rmul__ = __mul__

    def scale(self, c):
        return DIA(self.v.scale(c), self.d.scale(c))

    def sin(self):
        return DIA(_trig(self.v, False), _trig(self.v, True) * self.d)

    def cos(self):
        return DIA(_trig(self.v, True), -(_trig(self.v, False) * self.d))

    @property
    def shape(self):
        return self.v.shape
