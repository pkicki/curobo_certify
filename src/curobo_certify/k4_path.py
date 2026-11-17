"""K4 path-level continuous-time torque certificate for real cuRobo TimeOpt outputs (Panda).

Contract MODEL_K4 [CUROBO_PINNED_PANDA_RIGID_BODY_MODEL]: for the exact serialized path q(s)
(effective cubic geometry) and squared speed x(s) (32 cubic timing controls), on the common
q/x knot partition (spline_polynomials.extract_timeopt_spans, exact rationals), every joint torque
    tau(s) = A(q,q',q'') x(s) + B(q,q') x'(s) + G(q)
lies in [lower_j, upper_j] for EVERY s in [0, 1]. Proof: each span is split dyadically; on each
leaf, exact Bernstein hulls of q, q', q'', x, x' (outward to binary64) feed the interval RNEA
(k4_panda_model) and the interval sum must be inside the limits. VIOLATED_TORQUE requires a point
s (exact rational) whose rigorous point torque enclosure is disjoint from the limit interval.
Budget exhaustion is UNKNOWN_DEPTH / UNKNOWN_NODES. x(s) < 0 at a point is VIOLATED_X.

Not covered: K3 kinematic limits / positivity / endpoint obligations (separate certificate; the
combined K3+K4 status requires both), native FP32 torque, friction, motor limits, torque rate,
model physical fidelity (BLOCKED_MODEL_PHYSICAL_VALIDITY, see docs/K4_P0_MODEL_CONTRACT_20261117.md).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction as Q

import numpy as np

from .k4_interval import IA
from .k4_panda_model import PandaModel
from .k4_reference import bernstein, dpoly, eval_poly, split

SCOPE = "MODEL_K4_TORQUE[CUROBO_PINNED_PANDA_RIGID_BODY_MODEL]"


def _f_lo(v: Q) -> float:
    f = float(v)
    return f if Q(f) <= v else float(np.nextafter(f, -np.inf))


def _f_hi(v: Q) -> float:
    f = float(v)
    return f if Q(f) >= v else float(np.nextafter(f, np.inf))


@dataclass
class K4PathResult:
    status: str
    scope: str
    nodes: int
    max_depth: int
    leaves: list = field(default_factory=list)          # (span, path, lo[7], hi[7])
    witness: dict | None = None
    worst_utilization: list | None = None                # per joint max over leaves of |bound| / limit side
    seconds: float = 0.0
    reason: str = ""


def _raw_polys(sp, dof):
    h = sp.right - sp.left
    return ([tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q]
            + [tuple(sp.x), dpoly(sp.x, 1, h)])


def _boxes(bern_list, dof):
    """bern_list: list over leaves of tuples of Bernstein coefficient tuples -> IA arrays."""
    n = len(bern_list)
    k = 3 * dof + 2
    lo = np.empty((k, n))
    hi = np.empty((k, n))
    for i, b in enumerate(bern_list):
        for r in range(k):
            lo[r, i] = _f_lo(min(b[r]))
            hi[r, i] = _f_hi(max(b[r]))
    ia = [IA(lo[r], hi[r]) for r in range(k)]
    return ia[:dof], ia[dof:2 * dof], ia[2 * dof:3 * dof], ia[3 * dof], ia[3 * dof + 1]


def _point_boxes(raw_list, us, dof):
    n = len(us)
    k = 3 * dof + 2
    lo = np.empty((k, n))
    hi = np.empty((k, n))
    exact = []
    for i, (raw, u) in enumerate(zip(raw_list, us)):
        vals = [eval_poly(p, u) for p in raw]
        exact.append(vals)
        for r, v in enumerate(vals):
            lo[r, i], hi[r, i] = _f_lo(v), _f_hi(v)
    ia = [IA(lo[r], hi[r]) for r in range(k)]
    return ia[:dof], ia[dof:2 * dof], ia[2 * dof:3 * dof], ia[3 * dof], ia[3 * dof + 1], exact


def torque_enclosure(model: PandaModel, q, qp, qpp, x, xp):
    A, B, G = model.coefficients(q, qp, qpp)
    return [G[j] + A[j] * x + B[j] * xp for j in range(model.dof)]


def certify_path(spans, model: PandaModel, lower, upper, *, max_depth=12, max_nodes=65536,
                 batch=4096, keep_leaves=True) -> K4PathResult:
    t0 = time.perf_counter()
    dof = model.dof
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != (dof,) or upper.shape != (dof,) or not (lower < upper).all():
        return K4PathResult("INVALID_INPUT", SCOPE, 0, 0, reason="torque limits")
    if not spans or any(len(sp.q) != dof for sp in spans):
        return K4PathResult("INVALID_INPUT", SCOPE, 0, 0, reason="span dimension")
    raws = [_raw_polys(sp, dof) for sp in spans]
    active = [(si, "", Q(0), Q(1), tuple(bernstein(p) for p in raws[si])) for si in range(len(spans))]
    leaves, nodes, maxd = [], 0, 0
    worst = np.zeros(dof)
    while active:
        if nodes + len(active) > max_nodes:
            return K4PathResult("UNKNOWN_NODES", SCOPE, nodes, maxd, leaves, None, worst.tolist(),
                                time.perf_counter() - t0, "node budget")
        nxt = []
        for c0 in range(0, len(active), batch):
            chunk = active[c0:c0 + batch]
            nodes += len(chunk)
            q, qp, qpp, x, xp = _boxes([c[4] for c in chunk], dof)
            tau = torque_enclosure(model, q, qp, qpp, x, xp)
            tlo = np.stack([t.lo for t in tau], 1)
            thi = np.stack([t.hi for t in tau], 1)
            finite = np.isfinite(tlo).all(1) & np.isfinite(thi).all(1)
            ok = finite & (x.lo >= 0) & (tlo >= lower).all(1) & (thi <= upper).all(1)
            util = np.maximum(np.where(thi > 0, thi / upper, 0), np.where(tlo < 0, tlo / lower, 0))
            fail = [i for i in range(len(chunk)) if not ok[i]]
            for i in np.nonzero(ok)[0]:
                worst = np.maximum(worst, util[i])
                c = chunk[i]
                maxd = max(maxd, len(c[1]))
                if keep_leaves:
                    leaves.append((c[0], c[1], tlo[i].tolist(), thi[i].tolist()))
            if fail:
                # rigorous point witnesses at the leaf ends and midpoint (exact rational s)
                pts, raw_l, meta = [], [], []
                for i in fail:
                    si, path, a, b, _ = chunk[i]
                    for u in (a, (a + b) / 2, b):
                        pts.append(u)
                        raw_l.append(raws[si])
                        meta.append((si, path, u))
                pq, pqp, pqpp, px, pxp, exact = _point_boxes(raw_l, pts, dof)
                for k, (si, path, u) in enumerate(meta):
                    if exact[k][3 * dof] < 0:
                        return K4PathResult("VIOLATED_X", SCOPE, nodes, maxd, leaves,
                                            {"span": si, "u": str(u), "x": str(exact[k][3 * dof])}, worst.tolist(),
                                            time.perf_counter() - t0, "negative squared speed")
                ptau = torque_enclosure(model, pq, pqp, pqpp, px, pxp)
                plo = np.stack([t.lo for t in ptau], 1)
                phi = np.stack([t.hi for t in ptau], 1)
                for k, (si, path, u) in enumerate(meta):
                    for j in range(dof):
                        if np.isfinite(plo[k, j]) and np.isfinite(phi[k, j]) and (plo[k, j] > upper[j] or phi[k, j] < lower[j]):
                            s = spans[si].left + (spans[si].right - spans[si].left) * u
                            return K4PathResult("VIOLATED_TORQUE", SCOPE, nodes, maxd, leaves, {
                                "span": si, "u": str(u), "s": str(s), "s_float": float(s), "joint": j,
                                "torque_enclosure": [float(plo[k, j]), float(phi[k, j])],
                                "limit": [float(lower[j]), float(upper[j])],
                                "excess": float(plo[k, j] - upper[j]) if plo[k, j] > upper[j] else float(lower[j] - phi[k, j])},
                                worst.tolist(), time.perf_counter() - t0, "rigorous point witness")
                for i in fail:
                    si, path, a, b, bern = chunk[i]
                    if len(path) >= max_depth:
                        return K4PathResult("UNKNOWN_DEPTH", SCOPE, nodes, max(maxd, len(path)), leaves, None,
                                            worst.tolist(), time.perf_counter() - t0, "subdivision exhausted")
                    halves = [split(p) for p in bern]
                    mid = (a + b) / 2
                    nxt.append((si, path + "L", a, mid, tuple(h[0] for h in halves)))
                    nxt.append((si, path + "R", mid, b, tuple(h[1] for h in halves)))
        active = nxt
    return K4PathResult("CERTIFIED", SCOPE, nodes, maxd, leaves, None, worst.tolist(),
                        time.perf_counter() - t0, "every leaf torque enclosure inside limits")
