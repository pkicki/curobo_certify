"""Interval Bernstein polynomial operations; coefficients are ascending in index."""
from __future__ import annotations
from fractions import Fraction
from math import comb
from .interval import Interval, as_interval

def power_to_bernstein(power, degree=None):
    """Convert ascending power coefficients on [0,1] to Bernstein form."""
    raw = list(power)
    if not raw: raise ValueError("power polynomial must not be empty")
    n=max(len(raw)-1, 0) if degree is None else degree
    if n < len(raw)-1: raise ValueError("degree cannot truncate polynomial")
    # Exact point inputs admit an exact rational linear transform before the
    # final outward conversion. This preserves cancellations such as the zero
    # endpoint coefficient of u(1-u), which is needed for strict-rest proofs.
    point_values = []
    for value in raw:
        if isinstance(value, Interval):
            if value.lo != value.hi:
                point_values = None
                break
            point_values.append(Fraction.from_float(value.lo))
        elif isinstance(value, Fraction):
            point_values.append(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            point_values.append(Fraction(value))
        else:
            point_values.append(Fraction.from_float(float(value)))
    if point_values is not None:
        point_values += [Fraction(0)]*(n+1-len(point_values))
        return [Interval.rational(sum((point_values[i] * Fraction(comb(k,i),comb(n,i))
                                       for i in range(k+1)), Fraction(0)))
                for k in range(n+1)]
    a=[as_interval(x) for x in raw]
    a += [Interval.point(0)]*(n+1-len(a))
    return [sum((a[i] * Fraction(comb(k,i),comb(n,i)) for i in range(k+1)), Interval.point(0)) for k in range(n+1)]

def bernstein_to_power(coeff):
    """Convert Bernstein coefficients to ascending power coefficients."""
    b=[as_interval(x) for x in coeff]; n=len(b)-1
    out=[]
    for i in range(n+1):
        out.append(sum((b[k] * ((-1)**(i-k)*comb(n,i)*comb(i,k)) for k in range(i+1)), Interval.point(0)))
    return out

def derivative(coeff, span_length=1.0):
    """Global-coordinate derivative on a span of positive global length."""
    b=[as_interval(x) for x in coeff]; n=len(b)-1
    if not math_is_positive_finite(span_length): raise ValueError("span length must be positive and finite")
    if n == 0: return [Interval.point(0)]
    h = span_length if isinstance(span_length, Fraction) else Fraction.from_float(float(span_length))
    if h <= 0: raise ValueError("span length must be positive")
    return [(b[i+1]-b[i]) * (Fraction(n, 1) / h) for i in range(n)]

def product(a,b):
    """Bernstein product using binomial-weight convolution."""
    a=[as_interval(x) for x in a]; b=[as_interval(x) for x in b]
    m,n=len(a)-1,len(b)-1
    out=[]
    for k in range(m+n+1):
        terms=[]
        for i in range(max(0,k-n),min(m,k)+1):
            j=k-i
            terms.append(a[i]*b[j]*Fraction(comb(m,i)*comb(n,j),comb(m+n,k)))
        out.append(sum(terms,Interval.point(0)))
    return out

def elevate(coeff, target_degree):
    b=[as_interval(x) for x in coeff]
    if target_degree < len(b)-1: raise ValueError("degree elevation cannot lower degree")
    while len(b)-1 < target_degree:
        n=len(b)-1
        b=[b[0]]+[b[i-1]*Fraction(i,n+1)+b[i]*Fraction(n+1-i,n+1) for i in range(1,n+1)]+[b[-1]]
    return b

def split(coeff, t=Fraction(1,2)):
    """de Casteljau split, returned child coefficients enclose exact subdivision."""
    b=[as_interval(x) for x in coeff]
    t=as_interval(t); omt=Interval.point(1)-t
    rows=[b]
    while len(rows[-1])>1:
        prev=rows[-1]
        rows.append([omt*prev[i]+t*prev[i+1] for i in range(len(prev)-1)])
    return [row[0] for row in rows], [row[-1] for row in reversed(rows)]

def math_is_positive_finite(x):
    import math
    return math.isfinite(x) and x>0
