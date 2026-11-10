"""Small one-span CPU K3 reference certifier; no geometry extraction is implied."""
from __future__ import annotations
from dataclasses import dataclass, asdict
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
    leaves=[]; processed=0; any_unknown=False; any_violated=False
    stack=[(list(map(as_interval,q)),list(map(as_interval,x)),0,"")]
    while stack:
        qq,xx,depth,path=stack.pop(); processed+=1
        residuals=k3_residuals(qq,xx,limits,span_length/(2**depth))
        positive=all(c.lo>=0 for c in xx) and any(c.lo>0 for c in xx)
        upper={name:max(c.hi for c in coeff) for name,coeff in residuals.items()}
        safe=positive and all(u<=0 for u in upper.values())
        if safe:
            leaves.append(Leaf(path,depth,"CERTIFIED",upper,min(c.lo for c in xx))); continue
        if processed+len(stack)+2>max_nodes:
            any_unknown=True; leaves.append(Leaf(path,depth,"UNKNOWN_NODES",upper,min(c.lo for c in xx))); continue
        # Rigorous witness at midpoint: evaluate polynomial at u=1/2 by de Casteljau.
        violation=None
        for name, coeff in residuals.items():
            l,_=split(coeff)
            if l[-1].lo>0: violation=name; break
        if violation:
            any_violated=True; leaves.append(Leaf(path,depth,"VIOLATED:"+violation,upper,min(c.lo for c in xx))); continue
        if depth>=max_depth:
            any_unknown=True; leaves.append(Leaf(path,depth,"UNKNOWN_DEPTH",upper,min(c.lo for c in xx))); continue
        ql,qr=split(qq); xl,xr=split(xx)
        stack.append((qr,xr,depth+1,path+"1")); stack.append((ql,xl,depth+1,path+"0"))
    status=Status.VIOLATED.value if any_violated else ("UNKNOWN_NODES" if any(l.status=="UNKNOWN_NODES" for l in leaves) else ("UNKNOWN_DEPTH" if any_unknown else Status.UNKNOWN_BOUNDARY.value))
    return SpanResult(status,Status.UNSUPPORTED.value,"single-span residual/positivity proof only; endpoint-state and multi-span regularity evidence absent; torque/rate unsupported",processed,leaves)

def result_dict(result): return asdict(result)
