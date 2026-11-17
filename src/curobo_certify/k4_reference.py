"""Research K4 reference: conservative continuous-path inverse-dynamics torque bounds.

MODEL_TORQUE_ONLY for analytical 1/2-DOF *planar* rigid-body fixtures, NOT Panda,
not native FP32 torque, not motor-side torque and not certified physical execution.
All polynomial coefficients and model parameters are exact rational inputs.
Trig functions use a Taylor polynomial plus a proven Lagrange remainder and the
1-Lipschitz property of sin/cos; no sampled torque curve is used as a bound.

This is a deliberately small, independent correctness reference for the future
robot-general interval RNEA / certified Taylor-model implementation. A replay
verifier is provided for structural tests, but shares the dynamics equations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import comb, factorial, isfinite
from typing import Sequence


def rat(x):
    if isinstance(x, bool) or (isinstance(x, float) and not isfinite(x)):
        raise ValueError("nonfinite/boolean rational input")
    return Q(x)


@dataclass(frozen=True)
class I:
    lo: Q
    hi: Q

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("empty interval")

    @classmethod
    def point(cls, x):
        v = rat(x)
        return cls(v, v)

    @classmethod
    def of(cls, x):
        return x if isinstance(x, cls) else cls.point(x)

    def __add__(self, other):
        other = I.of(other)
        return I(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -I.of(other)

    def __rsub__(self, other):
        return I.of(other) - self

    def __mul__(self, other):
        other = I.of(other)
        p = (self.lo * other.lo, self.lo * other.hi,
             self.hi * other.lo, self.hi * other.hi)
        return I(min(p), max(p))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = I.of(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("interval division through zero")
        return self * I(1 / other.hi, 1 / other.lo)

    def subset(self, other):
        other = I.of(other)
        return other.lo <= self.lo and self.hi <= other.hi

    def as_json(self):
        return [str(self.lo), str(self.hi)]


def _sin_cos_midpoint(c: Q, *, cosine: bool, order: int = 28) -> I:
    """Maclaurin through degree 2*order(+1 for sine), plus Lagrange bound.

    The derivative of every order of sin/cos is bounded in magnitude by 1 on R,
    so the absolute Taylor remainder is <= |c|**(degree+1)/(degree+1)!.
    """
    d = 2 * order + (0 if cosine else 1)
    p = sum(((-1 if k % 2 else 1) * c ** (2*k + (0 if cosine else 1)) /
             factorial(2*k + (0 if cosine else 1)) for k in range(order + 1)), Q(0))
    r = abs(c) ** (d + 1) / factorial(d + 1)
    return I(max(Q(-1), p-r), min(Q(1), p+r))


def _trig(a: I, *, cosine: bool) -> I:
    mid = (a.lo + a.hi) / 2
    radius = (a.hi - a.lo) / 2
    # Correct but coarse fallback for very large angles: no unverified range reduction.
    if abs(mid) > 16:
        return I(Q(-1), Q(1))
    z = _sin_cos_midpoint(mid, cosine=cosine)
    # |sin'(q)|, |cos'(q)| <= 1; interval is valid across all stationary points.
    return I(max(Q(-1), z.lo - radius), min(Q(1), z.hi + radius))


def isin(a):
    return _trig(I.of(a), cosine=False)


def icos(a):
    return _trig(I.of(a), cosine=True)


def eval_poly(p, u):
    value = Q(0)
    for c in reversed(p):
        value = value * u + c
    return value


def dpoly(p, order, h):
    result = tuple(p)
    for _ in range(order):
        result = tuple(Q(i)*result[i]/h for i in range(1, len(result))) or (Q(0),)
    return result


def bernstein(p):
    n = len(p)-1
    return tuple(sum((p[i] * Q(comb(k, i), comb(n, i)) for i in range(k+1)), Q(0))
                 for k in range(n+1))


def split(b):
    levels = [tuple(b)]
    while len(levels[-1]) > 1:
        p = levels[-1]
        levels.append(tuple((p[i] + p[i+1])/2 for i in range(len(p)-1)))
    n = len(b)-1
    return (tuple(levels[i][0] for i in range(n+1)),
            tuple(levels[n-i][i] for i in range(n+1)))


def hull(b):
    return I(min(b), max(b))


def compose_subinterval(p, a: Q, b: Q):
    """Independent exact affine substitution p(a+(b-a)*v) for replay."""
    out = [Q(0)] * len(p)
    for n, c in enumerate(p):
        for k in range(n+1):
            out[k] += c * comb(n,k) * a**(n-k) * (b-a)**k
    return tuple(out)


@dataclass(frozen=True)
class PathSpan:
    left: Q
    right: Q
    q: tuple[tuple[Q, ...], ...]
    x: tuple[Q, ...]

    @classmethod
    def from_values(cls, left, right, q, x):
        a, b = rat(left), rat(right)
        if b <= a or not q or not x or len(x)>4 or any(not p or len(p)>4 for p in q):
            raise ValueError("invalid cubic spline span")
        return cls(a,b,tuple(tuple(map(rat,p)) for p in q),tuple(map(rat,x)))

    @classmethod
    def from_k3_span(cls, span):
        return cls.from_values(span.left,span.right,span.q,span.x)


@dataclass(frozen=True)
class Pendulum:
    inertia: Q
    mass: Q
    com: Q
    gravity: Q
    dof: int = 1

    def __post_init__(self):
        for key in ("inertia", "mass", "com", "gravity"):
            object.__setattr__(self, key, rat(getattr(self, key)))
        if self.inertia < 0 or self.mass <= 0 or self.com <= 0 or self.gravity < 0:
            raise ValueError("invalid rigid-body parameters")

    def coefficients(self, q, qp, qpp):
        if len(q) != 1:
            raise ValueError("model expects one joint")
        Iq = I.of(self.inertia)
        return ((Iq*qpp[0],), (Iq*qp[0]*Q(1,2),),
                (self.mass*self.com*self.gravity*isin(q[0]),))


@dataclass(frozen=True)
class PlanarTwoLink:
    m1: Q
    m2: Q
    l1: Q
    c1: Q
    c2: Q
    J1: Q
    J2: Q
    gravity: Q
    dof: int = 2

    def __post_init__(self):
        for key in ("m1", "m2", "l1", "c1", "c2", "J1", "J2", "gravity"):
            object.__setattr__(self, key, rat(getattr(self, key)))
        if any(v <= 0 for v in (self.m1,self.m2,self.l1,self.c1,self.c2)) or any(
                v < 0 for v in (self.J1,self.J2,self.gravity)):
            raise ValueError("invalid rigid-body parameters")

    def coefficients(self, q, qp, qpp):
        if len(q) != 2:
            raise ValueError("model expects two joints")
        q1,q2=q; d1,d2=qp; dd1,dd2=qpp
        c=icos(q2); s=isin(q2)
        z=self.m2*self.l1*self.c2
        M11=self.J1+self.J2+self.m1*self.c1**2+self.m2*(self.l1**2+self.c2**2)+2*z*c
        M12=self.J2+self.m2*self.c2**2+z*c
        M22=I.of(self.J2+self.m2*self.c2**2)
        cori=-z*s
        A1=M11*dd1+M12*dd2+cori*(2*d1*d2+d2*d2)
        A2=M12*dd1+M22*dd2-cori*d1*d1
        B1=(M11*d1+M12*d2)*Q(1,2)
        B2=(M12*d1+M22*d2)*Q(1,2)
        g1=self.gravity*((self.m1*self.c1+self.m2*self.l1)*icos(q1)+self.m2*self.c2*icos(q1+q2))
        g2=self.gravity*self.m2*self.c2*icos(q1+q2)
        return ((A1,A2),(B1,B2),(g1,g2))


def _torque_bounds(model, q, qp, qpp, x, xp):
    A,B,G=model.coefficients(q,qp,qpp)
    return tuple(G[i]+A[i]*x+B[i]*xp for i in range(model.dof))


def _roots(sp):
    h=sp.right-sp.left
    powers=tuple(sp.q)+tuple(dpoly(p,1,h) for p in sp.q)+tuple(
        dpoly(p,2,h) for p in sp.q)+(sp.x,dpoly(sp.x,1,h))
    return tuple(bernstein(p) for p in powers)


def _intervals(b, dof):
    i=iter(b)
    q=tuple(hull(next(i)) for _ in range(dof))
    qp=tuple(hull(next(i)) for _ in range(dof))
    qpp=tuple(hull(next(i)) for _ in range(dof))
    return q,qp,qpp,hull(next(i)),hull(next(i))


def _point_torque(sp,model,u):
    h=sp.right-sp.left
    q=tuple(I.point(eval_poly(p,u)) for p in sp.q)
    qp=tuple(I.point(eval_poly(dpoly(p,1,h),u)) for p in sp.q)
    qpp=tuple(I.point(eval_poly(dpoly(p,2,h),u)) for p in sp.q)
    return _torque_bounds(model,q,qp,qpp,I.point(eval_poly(sp.x,u)),
                          I.point(eval_poly(dpoly(sp.x,1,h),u)))


def _regularity(spans, dof):
    if not spans or any(len(s.q)!=dof or s.right<=s.left for s in spans):
        return False
    for left,right in zip(spans,spans[1:]):
        if left.right!=right.left:
            return False
        for d in range(3):
            for i in range(dof):
                if eval_poly(dpoly(left.q[i],d,left.right-left.left),Q(1))!=eval_poly(
                    dpoly(right.q[i],d,right.right-right.left),Q(0)):
                    return False
        for d in range(2):
            if eval_poly(dpoly(left.x,d,left.right-left.left),Q(1))!=eval_poly(
                dpoly(right.x,d,right.right-right.left),Q(0)):
                return False
    return True


def _limits(limits,dof):
    if len(limits)!=dof:
        raise ValueError("torque limits must match model dimension")
    out=tuple(I(rat(lo),rat(hi)) for lo,hi in limits)
    if any(v.lo>=v.hi for v in out):
        raise ValueError("torque lower must be smaller than upper")
    return out


@dataclass(frozen=True)
class Leaf:
    span: int
    path: str
    torque: tuple[I, ...]


@dataclass(frozen=True)
class K4Result:
    status: str
    scope: str
    nodes: int
    leaves: tuple[Leaf, ...]
    witness: dict | None = None
    reason: str = ""

    def to_json(self):
        return {"status":self.status,"scope":self.scope,"nodes":self.nodes,
                "reason":self.reason,
                "leaves":[{"span":v.span,"path":v.path,"torque":[z.as_json() for z in v.torque]}
                          for v in self.leaves],
                "witness":None if self.witness is None else {k:(str(v) if isinstance(v,Q) else
                    [z.as_json() for z in v] if isinstance(v,tuple) else v) for k,v in self.witness.items()}}


SCOPE="ANALYTIC_PLANAR_MODEL_TORQUE_ONLY_REQUIRES_SEPARATE_K3_AND_DYNAMICS_VALIDATION"


def certify_torque(spans: Sequence[PathSpan], model, limits, *, max_depth=12, max_nodes=8192):
    """Certify continuous-path inverse-dynamics torque; fail closed on budget.

    Proof is for exact rational serialized analytic 1/2-link dynamics only.
    K3 constraints, model uncertainty, friction, payload attachment, torque-speed
    envelopes and native CUDA numerics are NOT covered.
    """
    if not isinstance(max_depth,int) or max_depth<0 or not isinstance(max_nodes,int) or max_nodes<1:
        return K4Result("INVALID_INPUT",SCOPE,0,(),reason="invalid budgets")
    try:
        lim=_limits(limits,model.dof)
        if not _regularity(spans,model.dof):
            return K4Result("INVALID_PATH",SCOPE,0,(),reason="span or C2(q)/C1(x) mismatch")
        leaves=[]; n=0
        for si,sp in enumerate(spans):
            queue=[("",Q(0),Q(1),_roots(sp))]
            while queue:
                if n>=max_nodes:
                    return K4Result("UNKNOWN_NODES",SCOPE,n,tuple(leaves),reason="node budget")
                path,a,b,data=queue.pop(0);n+=1
                q,qp,qpp,x,xp=_intervals(data,model.dof)
                tb=_torque_bounds(model,q,qp,qpp,x,xp)
                if x.lo>=0 and all(v.subset(lim[i]) for i,v in enumerate(tb)):
                    leaves.append(Leaf(si,path,tb)); continue
                for u in (a,(a+b)/2,b):
                    xv=eval_poly(sp.x,u)
                    if xv<0:
                        return K4Result("VIOLATED_X",SCOPE,n,tuple(leaves),
                            {"span":si,"u":u,"x":xv},"negative squared speed")
                    if xv>=0:
                        pt=_point_torque(sp,model,u)
                        for j,v in enumerate(pt):
                            if v.lo>lim[j].hi or v.hi<lim[j].lo:
                                return K4Result("VIOLATED_TORQUE",SCOPE,n,tuple(leaves),
                                    {"span":si,"u":u,"joint":j,"torque":pt},"rigorous point witness")
                if len(path)>=max_depth:
                    return K4Result("UNKNOWN_DEPTH",SCOPE,n,tuple(leaves),reason="subdivision exhausted")
                first=tuple(split(p)[0] for p in data)
                second=tuple(split(p)[1] for p in data)
                mid=(a+b)/2
                queue.extend(((path+"L",a,mid,first),(path+"R",mid,b,second)))
        return K4Result("CERTIFIED",SCOPE,n,tuple(leaves),reason="all interval torque bounds enclosed")
    except (ValueError,TypeError,ZeroDivisionError,OverflowError) as ex:
        return K4Result("INVALID_INPUT",SCOPE,0,(),reason=str(ex))


def replay_torque_leaves(spans, model, limits, result):
    """Reference proof replay with independently constructed subinterval polynomials.

    NOT an independent implementation of rigid-body dynamics. Used to detect
    tampered, missing, overlapping and numerically invalid leaf certificates.
    """
    if result.status!="CERTIFIED" or not _regularity(spans,model.dof):
        return False
    try:
        lim=_limits(limits,model.dof)
        by_span={i:[] for i in range(len(spans))}
        for leaf in result.leaves:
            if leaf.span not in by_span or any(c not in "LR" for c in leaf.path):
                return False
            by_span[leaf.span].append(leaf)
        for si,sp in enumerate(spans):
            ll=by_span[si];paths=[r.path for r in ll]
            if not paths or len(paths)!=len(set(paths)):
                return False
            if sum((Q(1,2**len(p)) for p in paths),Q(0))!=1:
                return False
            if any(x!=y and y.startswith(x) for x in paths for y in paths):
                return False
            h=sp.right-sp.left
            raw=tuple(sp.q)+tuple(dpoly(p,1,h) for p in sp.q)+tuple(
                dpoly(p,2,h) for p in sp.q)+(sp.x,dpoly(sp.x,1,h))
            for leaf in ll:
                a,b=Q(0),Q(1)
                for ch in leaf.path:
                    c=(a+b)/2
                    if ch=="L":b=c
                    else:a=c
                bnd=tuple(bernstein(compose_subinterval(p,a,b)) for p in raw)
                q,qp,qpp,x,xp=_intervals(bnd,model.dof)
                torque=_torque_bounds(model,q,qp,qpp,x,xp)
                if x.lo<0 or any(not t.subset(lim[j]) for j,t in enumerate(torque)):
                    return False
                if len(leaf.torque)!=model.dof or any(not t.subset(leaf.torque[j]) for j,t in enumerate(torque)):
                    return False
        return True
    except (ValueError,TypeError,ZeroDivisionError,OverflowError,AttributeError):
        return False
