"""Small one-span CPU K3 reference certifier; no geometry extraction is implied."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from fractions import Fraction
from .bernstein import split
from .contracts import Status
from .interval import Interval, as_interval
from .residuals import k3_residuals

@dataclass
class Leaf:
    path: str
    depth: int
    status: str
    residual_upper: dict
    x_lower: float
    witness: dict | None = None

@dataclass
class SpanResult:
    k3_status: str
    full_status: str
    reason: str
    processed_nodes: int
    leaves: list[Leaf]

def certify_span(q, x, limits, max_depth=12, max_nodes=8192, span_length=1.0):
    """Certify a single polynomial span; torque/rate are always unsupported.

    Positivity is proved from the Bernstein convex hull: all coefficient lower
    bounds are nonnegative and at least one is positive on every open leaf.
    Thus x>0 in each open leaf, while endpoints may be zero. This does not handle
    stationary geometry, multiple-span knot continuity, or native geometry parity.
    """
    if len(q)<1 or len(x)<1 or max_depth<0 or max_nodes<1:
        return SpanResult(Status.INVALID_INPUT.value,Status.UNSUPPORTED.value,"invalid input/budget",0,[])
    leaves=[]; processed=0; any_unknown=False; any_violated=False; stationary_unknown=False
    stack=[(list(map(as_interval,q)),list(map(as_interval,x)),0,"",Fraction(0),Fraction(1))]
    while stack:
        qq,xx,depth,path,left,right=stack.pop(); processed+=1
        residuals=k3_residuals(qq,xx,limits,span_length/(2**depth))
        x_left,x_right=xx[0],xx[-1]
        if x_left.hi < 0 or x_right.hi < 0:
            any_violated=True
            side, value = ("left", -x_left) if x_left.hi < 0 else ("right", -x_right)
            leaves.append(Leaf(path,depth,"VIOLATED:positive_x",{},min(c.lo for c in xx),
                               {"residual":"positive_x","side":side,"lower":value.lo,"upper":value.hi}))
            continue
        xmid_left,_=split(xx)
        xmid=xmid_left[-1]
        if xmid.hi < 0:
            any_violated=True
            value=-xmid
            leaves.append(Leaf(path,depth,"VIOLATED:positive_x",{},min(c.lo for c in xx),
                               {"residual":"positive_x","side":"mid","lower":value.lo,"upper":value.hi}))
            continue
        interior_left_uncertain = left > 0 and x_left.lo <= 0
        interior_right_uncertain = right < 1 and x_right.lo <= 0
        if interior_left_uncertain or interior_right_uncertain:
            stationary_unknown=True
            leaves.append(Leaf(path,depth,"UNKNOWN_STATIONARY",{},min(c.lo for c in xx)))
            continue
        endpoint_positive=(left == 0 or x_left.lo > 0) and (right == 1 or x_right.lo > 0)
        positive=all(c.lo>=0 for c in xx) and any(c.lo>0 for c in xx)
        upper={name:max(c.hi for c in coeff) for name,coeff in residuals.items()}
        safe=positive and endpoint_positive and all(u<=0 for u in upper.values())
        if safe:
            leaves.append(Leaf(path,depth,"CERTIFIED",upper,min(c.lo for c in xx))); continue
        if positive and not endpoint_positive:
            stationary_unknown=True
            leaves.append(Leaf(path,depth,"UNKNOWN_STATIONARY",upper,min(c.lo for c in xx)))
            continue
        if processed+len(stack)+2>max_nodes:
            any_unknown=True; leaves.append(Leaf(path,depth,"UNKNOWN_NODES",upper,min(c.lo for c in xx))); continue
        # Rigorous witness at midpoint: evaluate polynomial at u=1/2 by de Casteljau.
        violation=None
        for name, coeff in residuals.items():
            l,_=split(coeff)
            if coeff[0].lo>0: violation=(name,"left",coeff[0]); break
            if coeff[-1].lo>0: violation=(name,"right",coeff[-1]); break
            if l[-1].lo>0: violation=(name,"mid",l[-1]); break
        if violation:
            name,side,value=violation
            any_violated=True
            leaves.append(Leaf(path,depth,"VIOLATED:"+name+":"+side,upper,min(c.lo for c in xx),
                               {"residual":name,"side":side,"lower":value.lo,"upper":value.hi}))
            continue
        if depth>=max_depth:
            any_unknown=True; leaves.append(Leaf(path,depth,"UNKNOWN_DEPTH",upper,min(c.lo for c in xx))); continue
        ql,qr=split(qq); xl,xr=split(xx); middle=(left+right)/2
        stack.append((qr,xr,depth+1,path+"1",middle,right)); stack.append((ql,xl,depth+1,path+"0",left,middle))
    status=(Status.VIOLATED.value if any_violated else
            ("UNKNOWN_STATIONARY" if stationary_unknown else
             ("UNKNOWN_NODES" if any(l.status=="UNKNOWN_NODES" for l in leaves) else
              ("UNKNOWN_DEPTH" if any_unknown else Status.UNKNOWN_BOUNDARY.value))))
    return SpanResult(status,Status.UNSUPPORTED.value,"single-span residual/positivity proof only; endpoint-state and multi-span regularity evidence absent; torque/rate unsupported",processed,leaves)

def result_dict(result): return asdict(result)
