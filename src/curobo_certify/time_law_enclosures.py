"""Exact-rational enclosures of the MATHEMATICAL TimeOpt time law.

For a strictly positive piecewise-polynomial x(s), T(s)=integral ds/sqrt(x(s)).
On a dyadic interval [a,b], its Bernstein hull L<=x<=U proves:
  (b-a)/sqrt(U) <= integral_a^b 1/sqrt(x) ds <= (b-a)/sqrt(L).
All sqrt enclosures here use integer isqrt and exact rational arithmetic.

This is NOT a bound on cuRobo's native time->progress Newton kernel.
Strict-rest x=0 endpoints are explicitly UNSUPPORTED, not silently clipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt

from .exact_k3_reference import bernstein_coeffs

SCOPE = "EXACT_SERIALIZED_POSITIVE_X_TIME_LAW_ONLY_NOT_NATIVE_INVERSION"


def inv_sqrt_bracket(x: F, decimal_digits: int = 24) -> tuple[F, F]:
    x = F(x)
    if x <= 0 or not 1 <= decimal_digits <= 80:
        raise ValueError("strictly positive rational x and 1..80 digits required")
    q = 10 ** decimal_digits
    n = x.numerator
    a = x.denominator * q*q
    z = isqrt(a // n)
    floor = F(z,q)
    ceil = floor if n*z*z == a else F(z+1,q)
    # Proof checks are integer comparisons, not a floating point reference.
    assert floor*floor*x <= 1 and ceil*ceil*x >= 1
    return floor,ceil


def power_to_bernstein(power: tuple[F,...]) -> tuple[F,...]:
    if not 1 <= len(power) <= 4:
        raise ValueError("expected cubic-or-lower exact polynomial")
    coeff = bernstein_coeffs(tuple(F(c) for c in power))
    return tuple(coeff)


def subdivide_at(b: tuple[F,...], t: F) -> tuple[tuple[F,...],tuple[F,...]]:
    if not 0 <= t <= 1:
        raise ValueError("split location must lie in [0,1]")
    rows=[tuple(b)]
    while len(rows[-1]) > 1:
        v=rows[-1]
        rows.append(tuple((1-t)*v[i]+t*v[i+1] for i in range(len(v)-1)))
    degree=len(b)-1
    return (tuple(row[0] for row in rows),
            tuple(rows[degree-i][i] for i in range(degree+1)))


def subdivide_half(b: tuple[F,...]) -> tuple[tuple[F,...],tuple[F,...]]:
    return subdivide_at(b,F(1,2))


@dataclass(frozen=True)
class Piece:
    left: F
    right: F
    x_bernstein: tuple[F,...]

    @classmethod
    def from_power(cls,left,right,power):
        a,b=F(left),F(right)
        if b<=a:
            raise ValueError("nonpositive path span")
        return cls(a,b,power_to_bernstein(tuple(F(c) for c in power)))


@dataclass(frozen=True)
class TimeBracket:
    lower: F
    upper: F
    status: str
    depth: int
    pieces: int
    scope: str = SCOPE

    @property
    def width(self):
        return self.upper-self.lower


def _local_integral(b: tuple[F,...], h:F, *, depth:int, digits:int) -> tuple[F,F]:
    if h < 0:
        raise ValueError("negative integration width")
    if not h:
        return F(0),F(0)
    if depth<0 or depth>16:
        raise ValueError("depth must be in [0,16]")
    leaves=[b]
    for _ in range(depth):
        leaves=[child for leaf in leaves for child in subdivide_half(leaf)]
    lo=hi=F(0)
    for leaf in leaves:
        lower_x,upper_x=min(leaf),max(leaf)
        if lower_x<=0:
            raise ValueError("UNKNOWN_POSITIVITY: nonpositive Bernstein lower bound")
        lower_recip,_=inv_sqrt_bracket(upper_x,digits)
        _,upper_recip=inv_sqrt_bracket(lower_x,digits)
        lo+=(h/len(leaves))*lower_recip
        hi+=(h/len(leaves))*upper_recip
    if not lo<=hi:
        raise AssertionError("invalid time enclosure")
    return lo,hi


def time_prefix(pieces: tuple[Piece,...], s:F, *, depth=6, digits=24) -> TimeBracket:
    if not pieces:
        raise ValueError("missing pieces")
    s=F(s)
    if not pieces[0].left<=s<=pieces[-1].right:
        raise ValueError("progress outside certified model domain")
    for p,q in zip(pieces,pieces[1:]):
        if p.right!=q.left:
            raise ValueError("gap or overlap between polynomial spans")
    lo=hi=F(0)
    touched=0
    for piece in pieces:
        if s<=piece.left:
            break
        fraction = min(F(1),(s-piece.left)/(piece.right-piece.left))
        if fraction<=0:
            continue
        part,_=subdivide_at(piece.x_bernstein,fraction)
        try:
            lower,upper=_local_integral(part,(piece.right-piece.left)*fraction,
                                         depth=depth,digits=digits)
        except ValueError as exc:
            if "UNKNOWN_POSITIVITY" in str(exc):
                return TimeBracket(lo,hi,"UNKNOWN_POSITIVITY",depth,touched)
            raise
        lo+=lower
        hi+=upper
        touched+=1
        if s<=piece.right:
            break
    return TimeBracket(lo,hi,"BOUNDED_EXACT_MODEL",depth,touched)


@dataclass(frozen=True)
class InverseBracket:
    lower_s: F
    upper_s: F
    status: str
    steps: int
    depth: int
    scope: str = SCOPE


def inverse_time_bracket(pieces: tuple[Piece,...], target_t:F, *,
                         max_steps=32, initial_depth=4, max_depth=10,
                         digits=24) -> InverseBracket:
    """Bracket inverse of exact T(s) by comparison against RIGOROUS T bounds.

    Returns UNKNOWN rather than guessing if integral bounds overlap the target.
    Never implies cuRobo native inverse correctness or a physical-time derivative.
    """
    target_t=F(target_t)
    if target_t<0 or max_steps<1 or not 0<=initial_depth<=max_depth<=16:
        raise ValueError("invalid time or inverse budget")
    begin,end=pieces[0].left,pieces[-1].right
    if target_t==0:
        return InverseBracket(begin,begin,"BRACKETED_EXACT_MODEL",0,initial_depth)
    # The upper progress endpoint must be proved to lie AFTER the requested time.
    end_ready=False
    used_depth=initial_depth
    for depth in range(initial_depth,max_depth+1):
        bound=time_prefix(pieces,end,depth=depth,digits=digits)
        if bound.status!="BOUNDED_EXACT_MODEL":
            return InverseBracket(begin,end,bound.status,0,depth)
        if bound.lower>=target_t:
            end_ready=True
            used_depth=depth
            break
    if not end_ready:
        return InverseBracket(begin,end,"UNKNOWN_END_TIME",0,max_depth)
    low,high=begin,end
    for step in range(max_steps):
        mid=(low+high)/2
        decisive=False
        for depth in range(initial_depth,max_depth+1):
            trial=time_prefix(pieces,mid,depth=depth,digits=digits)
            used_depth=max(used_depth,depth)
            if trial.status!="BOUNDED_EXACT_MODEL":
                return InverseBracket(low,high,trial.status,step,depth)
            if trial.lower==trial.upper==target_t:
                return InverseBracket(mid,mid,"BRACKETED_EXACT_MODEL",step+1,depth)
            if trial.upper < target_t:
                low=mid;decisive=True;break
            if trial.lower > target_t:
                high=mid;decisive=True;break
        if not decisive:
            return InverseBracket(low,high,"UNKNOWN_TIME_BRACKET_OVERLAP",step+1,max_depth)
    return InverseBracket(low,high,"BRACKETED_EXACT_MODEL",max_steps,used_depth)
