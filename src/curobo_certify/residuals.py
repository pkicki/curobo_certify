"""K3 polynomial residual construction on one common normalized span."""
from __future__ import annotations
from fractions import Fraction
from .bernstein import derivative, elevate, product
from .interval import Interval, as_interval

def _add(a,b):
    n=max(len(a),len(b))-1; a=elevate(a,n); b=elevate(b,n)
    return [x+y for x,y in zip(a,b)]

def _scale(a,c): return [x*c for x in a]
def _minus_one(a): return _add(a,[Interval.point(-1)])
def _square(a): return product(a,a)

def k3_residuals(q, x, limits, span_length=1.0):
    """Return velocity, +/- acceleration, jerk residual Bernstein coefficients.

    q and x are ascending Bernstein coefficients on the same local u in [0,1].
    Limits are positive finite symmetric magnitudes. TimeOpt's x is squared speed.
    """
    import math
    if not q or not x or len(limits) != 3:
        raise ValueError("nonempty polynomials and positive finite limits required")
    bounds = tuple(as_interval(v) for v in limits)
    if any(not math.isfinite(v.lo) or not math.isfinite(v.hi) or v.lo <= 0 for v in bounds):
        raise ValueError("nonempty polynomials and positive finite limits required")
    vlim, alim, jlim = bounds
    q1=derivative(q,span_length); q2=derivative(q1,span_length); q3=derivative(q2,span_length)
    x1=derivative(x,span_length); x2=derivative(x1,span_length)
    velocity=_minus_one([c / vlim.square() for c in product(_square(q1),x)])
    acc=_add(product(q2,x),_scale(product(q1,x1),Fraction(1, 2)))
    accel_pos=_minus_one([c / alim for c in acc])
    accel_neg=_minus_one([c / alim for c in _scale(acc,-1)])
    ell=_add(_add(product(q3,x),_scale(product(q2,x1),Fraction(3, 2))),_scale(product(q1,x2),Fraction(1, 2)))
    jerk=_minus_one([c / jlim.square() for c in product(x,_square(ell))])
    return {"velocity":velocity,"acceleration_positive":accel_pos,"acceleration_negative":accel_neg,"jerk":jerk}
