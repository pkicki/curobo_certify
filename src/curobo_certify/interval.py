"""Outward-rounded binary64 intervals for the CPU reference backend.

Assumptions: IEEE-754 binary64 operations use round-to-nearest, ties-to-even;
``nextafter`` returns the adjacent representable value; Python is not compiled
with fast-math/reassociation; inputs denote their exact binary64 values. Every
primitive arithmetic result is expanded by one representable number. NaNs are
rejected; indeterminate extended-real operations conservatively return the
entire real line. This is not a proof for nonconforming arithmetic platforms.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math

_NEG_INF = float("-inf")
_POS_INF = float("inf")

def _down(x: float) -> float:
    return math.nextafter(x, _NEG_INF)

def _up(x: float) -> float:
    return math.nextafter(x, _POS_INF)

@dataclass(frozen=True, slots=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if math.isnan(self.lo) or math.isnan(self.hi) or self.lo > self.hi:
            raise ValueError("interval endpoints must be ordered and not NaN")

    @classmethod
    def point(cls, x: float) -> "Interval":
        x = float(x)
        if not math.isfinite(x):
            raise ValueError("point input must be finite")
        return cls(x, x)

    @classmethod
    def rational(cls, x: Fraction) -> "Interval":
        """Enclose an exact rational by adjacent binary64 values."""
        f = float(x)
        if not math.isfinite(f):
            if x > 0: return cls(math.nextafter(_POS_INF, 0.0), _POS_INF)
            return cls(_NEG_INF, math.nextafter(_NEG_INF, 0.0))
        represented = Fraction.from_float(f)
        if represented == x: return cls(f, f)
        if represented < x: return cls(f, _up(f))
        return cls(_down(f), f)

    def __add__(self, other):
        o = as_interval(other)
        x, y = self.lo + o.lo, self.hi + o.hi
        if math.isnan(x) or math.isnan(y): return Interval(_NEG_INF, _POS_INF)
        return Interval(_down(x), _up(y))
    __radd__ = __add__

    def __neg__(self): return Interval(-self.hi, -self.lo)
    def __sub__(self, other): return self + (-as_interval(other))
    def __rsub__(self, other): return as_interval(other) - self

    def __mul__(self, other):
        o = as_interval(other)
        vals = [a*b for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        if any(math.isnan(v) for v in vals): return Interval(_NEG_INF,_POS_INF)
        return Interval(_down(min(vals)), _up(max(vals)))
    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi: raise ZeroDivisionError("interval divisor contains zero")
        vals = (1.0/self.lo, 1.0/self.hi)
        return Interval(_down(min(vals)), _up(max(vals)))

    def __truediv__(self, other): return self * as_interval(other).reciprocal()
    def __rtruediv__(self, other): return as_interval(other) / self

    def square(self):
        if self.lo <= 0 <= self.hi:
            return Interval(0.0, _up(max(self.lo*self.lo, self.hi*self.hi)))
        vals=(self.lo*self.lo,self.hi*self.hi)
        return Interval(_down(min(vals)),_up(max(vals)))

    def contains(self, x: float) -> bool: return self.lo <= x <= self.hi
    def strictly_negative(self) -> bool: return self.hi < 0
    def strictly_positive(self) -> bool: return self.lo > 0

def as_interval(value) -> Interval:
    if isinstance(value, Interval): return value
    if isinstance(value, Fraction): return Interval.rational(value)
    return Interval.point(value)
