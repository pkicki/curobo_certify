"""P2 falsification campaign: reviewed Panda certifier vs the independent Lagrangian oracle.

Reviewed code is only CALLED (PandaModel, WorldFrameDynamics, certify_path, verify_k4,
verify_k4_witness); all truth values come from independent_checks.k4p2.oracle.
"""
from __future__ import annotations

import json
import random
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from . import oracle as O

ROOT = Path(__file__).resolve().parents[2]
TOL_MP = Q(1, 10 ** 30)        # mp point oracle (160-bit) accuracy margin; iv oracle uses 0


def links_from_contract(contract):
    """Reviewer's own reading of contract v2 JSON into oracle links (used when the URDF is absent)."""
    out = []
    for e in contract["model"]["links"]:
        out.append(O.OLink(e["name"], int(e["parent"]), int(e["joint_index"]),
                           tuple(Q(v) for v in e["origin_xyz"]), tuple(Q(v) for v in e["origin_rpy"]),
                           Q(e["mass"]), tuple(Q(v) for v in e["com"]), tuple(Q(v) for v in e["inertia"])))
    return out, tuple(Q(v) for v in contract["model"]["gravity"])


def oracle(kind="iv", contract=None, prec=160):
    if O.URDF.exists():
        links, g = O.parse_urdf_model()
    else:
        links, g = links_from_contract(contract)
    return O.LagrangeOracle(links, g, O.Ctx(kind, prec))


def _tol(orc):
    return Q(0) if orc.ctx.kind == "iv" else TOL_MP


def _count(st, v):
    st[v] = st.get(v, 0) + 1


def check_states(orc, M, W, rng, n, *, ranges=(-2.9, 2.9)):
    """Random exact (float64) states: producer RNEA torque + A/B/G, verifier world-frame torque."""
    from curobo_certify.k4_interval import IA
    st, bad = {}, []
    K = orc.ctx.m.mpf
    for _ in range(n):
        q = rng.uniform(*ranges, 7)
        v = rng.uniform(-3, 3, 7) * rng.choice([0.0, 1.0, 3.0])
        a = rng.uniform(-10, 10, 7)
        tq = orc.inverse_dynamics([K(x) for x in q], [K(x) for x in v], [K(x) for x in a])
        A, B, G = orc.abg([K(x) for x in q], [K(x) for x in v], [K(x) for x in a])
        bx = lambda arr: [IA(x) for x in arr]  # noqa: E731
        Pt = M.torque(bx(q), bx(v), bx(a))
        PA, PB, PG = M.coefficients(bx(q), bx(v), bx(a))
        Wt = W.torque(bx(q), bx(v), bx(a), True)
        for name, prod, ora in (("tau_rnea", Pt, tq), ("A", PA, A), ("B", PB, B), ("G", PG, G),
                                ("tau_world", Wt, tq)):
            for j in range(7):
                vd = O.verdict(prod[j].lo, prod[j].hi, ora[j], _tol(orc))
                if vd == "AMBIGUOUS" and name == "B" and not v.any() and prod[j].lo <= 0 <= prod[j].hi:
                    vd = "EXACT_ZERO_CONTAINED"   # v == 0 exactly => B = M(q) 0 / 2 = 0 exactly
                _count(st, vd)
                if vd not in ("CONTAINED", "EXACT_ZERO_CONTAINED"):
                    bad.append({"what": name, "joint": j, "verdict": vd, "q": q.tolist(), "v": v.tolist(),
                                "a": a.tolist(), "box": [float(prod[j].lo), float(prod[j].hi)],
                                "oracle": [float(x) for x in O.iv_bounds(ora[j])]})
    return st, bad


def check_boxes(orc, M, W, rng, nboxes, samples, *, radii=(1e-6, 1e-3, 1e-2, 5e-2)):
    """Random sub-boxes: producer/verifier enclosures vs dense oracle samples (corners + random)."""
    from curobo_certify.k4_interval import IA
    st, bad = {}, []
    K = orc.ctx.m.mpf
    for _ in range(nboxes):
        r = float(rng.choice(radii))
        cq, cv, ca = rng.uniform(-2.8, 2.8, 7), rng.uniform(-2, 2, 7), rng.uniform(-6, 6, 7)
        x0, xr, xp0, xpr = float(rng.uniform(0, 5)), r * 10, float(rng.uniform(-5, 5)), r * 10
        boxes = lambda c, rr: [IA(x - rr, x + rr) for x in c]  # noqa: E731
        qB, vB, aB = boxes(cq, r), boxes(cv, r), boxes(ca, r)
        xB, xpB = IA(x0, x0 + xr), IA(xp0 - xpr, xp0 + xpr)
        PA, PB, PG = M.coefficients(qB, vB, aB)
        Ptau = [PG[j] + PA[j] * xB + PB[j] * xpB for j in range(7)]
        Wtau = W.leaf_torque(qB, vB, aB, xB, xpB)
        for k in range(samples):
            if k < 2:   # opposite corners
                sgn = 1.0 if k == 0 else -1.0
                q, v, a = cq + sgn * r, cv - sgn * r, ca + sgn * r
                x, xp = (x0 + xr if k == 0 else x0), xp0 + sgn * xpr
            else:
                q, v, a = (c + rng.uniform(-r, r, 7) for c in (cq, cv, ca))
                x, xp = x0 + float(rng.uniform(0, xr)), xp0 + float(rng.uniform(-xpr, xpr))
            A, B, G = orc.abg([K(z) for z in q], [K(z) for z in v], [K(z) for z in a])
            # torque at this sample via the physical substitution (qd = v sqrt x, qdd = a x + v x'/2)
            w = orc.ctx.sqrt(K(x))
            tq = orc.inverse_dynamics([K(z) for z in q], [K(z) * w for z in v],
                                      [K(aa) * K(x) + K(vv) * K(xp) / 2 for aa, vv in zip(a, v)])
            for name, prod, ora in (("A", PA, A), ("B", PB, B), ("G", PG, G), ("tau_producer", Ptau, tq),
                                    ("tau_verifier", Wtau, tq)):
                for j in range(7):
                    vd = O.verdict(prod[j].lo, prod[j].hi, ora[j], _tol(orc))
                    _count(st, vd)
                    if vd != "CONTAINED":
                        bad.append({"what": name, "joint": j, "verdict": vd, "radius": r,
                                    "box": [float(prod[j].lo), float(prod[j].hi)],
                                    "oracle": [float(x_) for x_ in O.iv_bounds(ora[j])]})
    return st, bad


def leaf_interval(path):
    a = Q(0)
    for i, ch in enumerate(path):
        if ch == "R":
            a += Q(1, 2 ** (i + 1))
        elif ch != "L":
            raise ValueError(ch)
    return a, a + Q(1, 2 ** len(path))


def check_path_result(orc, spans, res, lower, upper, rng, samples):
    """Every reviewed claim in a K4PathResult against the oracle."""
    st, bad = {}, []
    if res.status == "CERTIFIED":
        # coverage of every span by the claimed leaves (own partition check)
        by = {}
        for (si, path, lo, hi) in res.leaves:
            a, b = leaf_interval(path)
            by.setdefault(si, []).append((a, b, lo, hi))
        for si in range(len(spans)):
            ivs = sorted((a, b) for a, b, _, _ in by.get(si, []))
            cur = Q(0)
            for a, b in ivs:
                if a != cur:
                    bad.append({"what": "partition gap/overlap", "span": si})
                    break
                cur = b
            if cur != 1:
                bad.append({"what": "partition incomplete", "span": si})
        for _ in range(samples):
            si, path, lo, hi = res.leaves[rng.randrange(len(res.leaves))]
            a, b = leaf_interval(path)
            u = rng.choice([a, b, (a + b) / 2, a + (b - a) * Q(rng.randint(1, 10 ** 6 - 1), 10 ** 6)])
            tq = O.path_torque(orc, spans[si], u)
            for j in range(7):
                vd = O.verdict(lo[j], hi[j], tq[j], _tol(orc))
                _count(st, vd)
                if vd != "CONTAINED":
                    bad.append({"what": "leaf box", "span": si, "u": str(u), "joint": j, "verdict": vd})
                lv = O.verdict(lower[j], upper[j], tq[j], _tol(orc))
                if lv == "DISJOINT":
                    bad.append({"what": "CERTIFIED but oracle torque outside limit", "span": si, "u": str(u)})
    if res.status == "VIOLATED_TORQUE":
        w = res.witness
        tq = O.path_torque(orc, spans[w["span"]], Q(w["u"]))[w["joint"]]
        a, b = O.iv_bounds(tq)
        tol = _tol(orc)
        ok = (a - tol > Q(float(upper[w["joint"]]))) or (b + tol < Q(float(lower[w["joint"]])))
        _count(st, "WITNESS_CONFIRMED" if ok else "WITNESS_REFUTED")
        if not ok:
            bad.append({"what": "false witness", **{k: str(v) for k, v in w.items()}})
    if res.status == "VIOLATED_X":
        w = res.witness
        x = O.peval(O.path_raw(spans[w["span"]])[3], Q(w["u"]))
        _count(st, "X_WITNESS_CONFIRMED" if x < 0 else "X_WITNESS_REFUTED")
        if x >= 0:
            bad.append({"what": "false x witness", "u": w["u"]})
    return st, bad


def fixture_spans(root=ROOT):
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    fx = json.loads((root / "fixtures/k4_real_path_margin0000.json").read_text())
    geo = np.array([[float.fromhex(h) for h in r] for r in fx["effective_geometry_hex"]], dtype=np.float32)
    tc = np.array([float.fromhex(h) for h in fx["timing_controls_hex"]], dtype=np.float32)
    return extract_timeopt_spans(geo, tc)


def synthetic_spans(rng: random.Random, nspans=2, x_kind="pos"):
    """Random rational cubic Panda paths (C-inf across knots by construction: one global cubic)."""
    from curobo_certify.boundary_reference import Span
    knots = [Q(0)] + sorted({Q(rng.randint(1, 15), 16) for _ in range(nspans - 1)}) + [Q(1)]
    P = [[Q(rng.randint(-1600, 1600), 1000)] + [Q(rng.randint(-800, 800), 1000) for _ in range(3)] for _ in range(7)]
    if x_kind == "pos":
        X = [Q(rng.randint(100, 4000), 1000), Q(rng.randint(-1000, 1000), 1000), Q(rng.randint(-1000, 1000), 1000), Q(0)]
        X[0] += sum(abs(c) for c in X[1:])
    elif x_kind == "zero_start":
        X = [Q(0), Q(rng.randint(100, 2000), 1000), Q(rng.randint(0, 1000), 1000), Q(0)]
    elif x_kind == "neg_pocket":   # x = (s - 5/16)^2 - eps : negative only near non-dyadic? 5/16 is dyadic
        c, eps = Q(1, 3), Q(1, 10 ** 9)
        X = [c * c - eps, -2 * c, Q(1), Q(0)]
    else:
        X = [Q(0)] * 4
    out = []
    for l, r in zip(knots, knots[1:]):
        h = r - l
        sh = lambda p: tuple(sum(Q(p[n]) * __import__("math").comb(n, k) * l ** (n - k) * h ** k  # noqa: E731
                                 for n in range(k, len(p))) for k in range(len(p)))
        out.append(Span(l, r, tuple(sh(p) for p in P), sh(X)))
    return out
