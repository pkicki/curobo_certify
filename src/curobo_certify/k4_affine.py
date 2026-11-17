"""K4 geometry-once, retime-many reference: interval-affine inverse-dynamics boxes.

Prepared coefficient boxes A(s),B(s),G(s) enclose tau=A*x+B*x'+G
for a *frozen analytical planar geometry*, and are independent of x(s).
The path hash, model identity and every interval box are verified before use.
This is an exact-rational correctness prototype; not fast Panda RNEA and not
independently dynamics-verified (same analytical models as k4_reference).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction as Q

from .k4_reference import (I, PathSpan, Leaf, K4Result, SCOPE, _limits,
                           _regularity, bernstein, compose_subinterval,
                           dpoly, hull, eval_poly)


def geometry_digest(spans):
    source = repr(tuple((s.left, s.right, s.q) for s in spans)).encode()
    return hashlib.sha256(source).hexdigest()


def model_digest(model):
    return hashlib.sha256(repr(model).encode()).hexdigest()


@dataclass(frozen=True)
class GeometryBox:
    span: int
    path: str
    A: tuple[I, ...]
    B: tuple[I, ...]
    G: tuple[I, ...]


@dataclass(frozen=True)
class PreparedGeometry:
    geometry_sha256: str
    model_sha256: str
    leaves: tuple[GeometryBox, ...]
    depth: int


def _coeff_box(sp: PathSpan, model, path):
    a,b=Q(0),Q(1)
    for step in path:
        c=(a+b)/2
        if step=="L":b=c
        else:a=c
    h=sp.right-sp.left
    q=tuple(hull(bernstein(compose_subinterval(p,a,b))) for p in sp.q)
    qp=tuple(hull(bernstein(compose_subinterval(dpoly(p,1,h),a,b))) for p in sp.q)
    qpp=tuple(hull(bernstein(compose_subinterval(dpoly(p,2,h),a,b))) for p in sp.q)
    return model.coefficients(q,qp,qpp)


def prepare_geometry(spans,model,depth=3,max_leaves=16384):
    if not isinstance(depth,int) or not 0<=depth<=14 or len(spans)*2**depth>max_leaves:
        raise ValueError("invalid geometry preparation budget")
    if not _regularity(spans,model.dof):
        raise ValueError("invalid path topology")
    leaves=[]
    for si,sp in enumerate(spans):
        for k in range(2**depth):
            path="".join("R" if (k>>i)&1 else "L" for i in reversed(range(depth)))
            A,B,G=_coeff_box(sp,model,path)
            leaves.append(GeometryBox(si,path,A,B,G))
    return PreparedGeometry(geometry_digest(spans),model_digest(model),tuple(leaves),depth)


def verify_prepared_geometry(spans,model,prepared):
    """Recompute all geometry boxes before treating cached data as a proof input."""
    if not _regularity(spans,model.dof):
        return False
    if prepared.geometry_sha256!=geometry_digest(spans) or prepared.model_sha256!=model_digest(model):
        return False
    seen=set()
    if len(prepared.leaves)!=len(spans)*(2**prepared.depth):
        return False
    for leaf in prepared.leaves:
        if not 0<=leaf.span<len(spans) or len(leaf.path)!=prepared.depth or any(c not in "LR" for c in leaf.path):
            return False
        if (leaf.span,leaf.path) in seen:
            return False
        seen.add((leaf.span,leaf.path))
        try:
            A,B,G=_coeff_box(spans[leaf.span],model,leaf.path)
        except (ValueError,ZeroDivisionError,TypeError):
            return False
        for recorded,exact in zip((leaf.A,leaf.B,leaf.G),(A,B,G)):
            if len(recorded)!=model.dof or any(not exact[i].subset(recorded[i]) for i in range(model.dof)):
                return False
    return True


def _certify_with_prepared_geometry(spans,model,limits,prepared):
    """Conservative certificate for a new timing law using verified frozen geometry.

    Returns UNKNOWN for inconclusive interval bounds; no claim about arbitrary
    online latency. Verifier checks the geometry coefficient boxes every call.
    """
    try:
        lim=_limits(limits,model.dof)
        if not verify_prepared_geometry(spans,model,prepared):
            return K4Result("INVALID_PREPARED_GEOMETRY",SCOPE,0,(),reason="invalid cached dynamics boxes")
        leaves=[]
        for leaf in prepared.leaves:
            sp=spans[leaf.span]
            a,b=Q(0),Q(1)
            for step in leaf.path:
                c=(a+b)/2
                if step=="L":b=c
                else:a=c
            x=hull(bernstein(compose_subinterval(sp.x,a,b)))
            xp=hull(bernstein(compose_subinterval(dpoly(sp.x,1,sp.right-sp.left),a,b)))
            torque=tuple(leaf.G[i]+leaf.A[i]*x+leaf.B[i]*xp for i in range(model.dof))
            if x.lo<0 or any(not v.subset(lim[i]) for i,v in enumerate(torque)):
                return K4Result("UNKNOWN_PREPARED_ENCLOSURE",SCOPE,len(leaves)+1,tuple(leaves),
                                reason="conservative geometry/timing box cannot establish limits")
            leaves.append(Leaf(leaf.span,leaf.path,torque))
        return K4Result("CERTIFIED",SCOPE,len(leaves),tuple(leaves),
                        reason="verified geometry coefficient boxes + timing Bernstein hulls")
    except (ValueError,ZeroDivisionError,TypeError,AttributeError) as exc:
        return K4Result("INVALID_INPUT",SCOPE,0,(),reason=str(exc))


def certify_with_prepared_geometry(spans,model,limits,prepared):
    """Wrapper setting the input digest on every result (P2 re-review R1): replay is strict about digests."""
    import dataclasses
    from .k4_reference import input_digest
    res = _certify_with_prepared_geometry(spans, model, limits, prepared)
    try:
        return dataclasses.replace(res, input_digest=input_digest(spans, model, limits))
    except (ValueError, TypeError, AttributeError):
        return res
