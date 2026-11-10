"""Separate checker for a one-span K3 leaf partition.

The checker derives each certified leaf polynomial from the original input by
replaying its binary path. It verifies prefix-free complete coverage, positivity,
and recomputed interval residual upper bounds. It does not trust the solver's
leaf bounds. This remains subject to the CPU interval assumptions document.
"""
from __future__ import annotations
from fractions import Fraction
from .bernstein import split
from .interval import as_interval
from .residuals import k3_residuals

def verify_span(q, x, limits, result, span_length=1.0):
    if result.k3_status not in ("CERTIFIED", "UNKNOWN_BOUNDARY") or not result.leaves:
        return False, "result is not a complete certified leaf set"
    paths=[leaf.path for leaf in result.leaves]
    if len(set(paths)) != len(paths): return False, "duplicate leaf path"
    if any(a != b and b.startswith(a) for a in paths for b in paths): return False, "overlapping leaf paths"
    # A finite prefix-free binary set covers [0,1] iff its Kraft sum is one.
    if sum(2.0**(-len(p)) for p in paths) != 1.0: return False, "leaf paths do not cover the span"
    root_q=list(map(as_interval,q)); root_x=list(map(as_interval,x))
    for leaf in result.leaves:
        if leaf.status != "CERTIFIED" or leaf.depth != len(leaf.path): return False, "invalid certified leaf record"
        qq,xx=root_q,root_x
        left,right=Fraction(0),Fraction(1)
        for bit in leaf.path:
            ql,qr=split(qq); xl,xr=split(xx)
            middle=(left+right)/2
            if bit=="0":
                qq,xx,right=ql,xl,middle
            elif bit=="1":
                qq,xx,left=qr,xr,middle
            else:
                return False, "invalid subdivision path"
        if not (all(c.lo>=0 for c in xx) and any(c.lo>0 for c in xx)):
            return False, f"positivity not proved at leaf {leaf.path}"
        if (left>0 and xx[0].lo<=0) or (right<1 and xx[-1].lo<=0):
            return False, f"interior stationary endpoint not excluded at leaf {leaf.path}"
        residual=k3_residuals(qq,xx,limits,span_length/(2**leaf.depth))
        for name,coeff in residual.items():
            if max(c.hi for c in coeff)>0: return False, f"residual fails at leaf {leaf.path}: {name}"
    return True, "all leaves replayed"


def verify_violation(q, x, limits, result, span_length=1.0):
    """Replay a rigorous point witness from a `VIOLATED` span result."""
    if result.k3_status != "VIOLATED":
        return False, "result does not report a rigorous violation"
    root_q=list(map(as_interval,q)); root_x=list(map(as_interval,x))
    for leaf in result.leaves:
        witness=leaf.witness
        if not leaf.status.startswith("VIOLATED:") or not isinstance(witness,dict):
            continue
        qq,xx=root_q,root_x
        for bit in leaf.path:
            ql,qr=split(qq); xl,xr=split(xx)
            if bit=="0": qq,xx=ql,xl
            elif bit=="1": qq,xx=qr,xr
            else: return False, "invalid witness subdivision path"
        side=witness.get("side")
        family=witness.get("residual")
        if family=="positive_x":
            candidates={"left":-xx[0],"right":-xx[-1],"mid":-split(xx)[0][-1]}
        else:
            residual=k3_residuals(qq,xx,limits,span_length/(2**leaf.depth))
            if family not in residual: return False, "unknown residual family in witness"
            coefficients=residual[family]
            candidates={"left":coefficients[0],"right":coefficients[-1],
                        "mid":split(coefficients)[0][-1]}
        value=candidates.get(side)
        if value is None or value.lo <= 0:
            continue
        if value.lo != witness.get("lower") or value.hi != witness.get("upper"):
            return False, "serialized witness enclosure differs from replay"
        return True, f"replayed positive {family} residual at leaf {leaf.path or 'root'}:{side}"
    return False, "no independently replayable positive witness found"
