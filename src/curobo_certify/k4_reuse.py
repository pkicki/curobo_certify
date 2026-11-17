"""K4 P3: certified geometry-once / retime-many torque enclosures for a frozen Panda path.

For a fixed geometric path q(s) (exact spans) and frozen model contract, the coefficient functions
A(s), B(s), G(s) do not depend on timing. This module caches, per (span, dyadic leaf):
  mode "box": rigorous constant enclosures [A], [B], [G] over the leaf (interval RNEA on Bernstein boxes);
  mode "tm1": a first-order Taylor model  C(s) in C(s_c) + [C'] (s - s_c),  s_c the leaf centre, where
              C(s_c) is a point-box interval RNEA and [C'] encloses dC/ds over the whole leaf via forward-mode
              dual intervals (k4_interval.DIA) through the same RNEA (needs q''' on the leaf).
For any squared-speed profile x(s) on the SAME geometry (new timing controls), torque on a leaf is
  box: [G] + [A] x([leaf]) + [B] x'([leaf])
  tm1: hull of the interval-coefficient Bernstein polynomial G0 + A0 x(s) + B0 x'(s)
       + ([G'] + [A'] x([leaf]) + [B'] x'([leaf])) * [-H, H],   H = leaf half-width in s.
Both are sufficient (sound) bounds; leaves that fail are refined (shared across timings, cached).
Witnesses and UNKNOWN follow k4_path. The cache is keyed by geometry SHA + contract SHA; a different
geometry, model or payload is rejected. Torque is model torque only (K4 scope as in k4_path).
"""
from __future__ import annotations

import hashlib
import time
from fractions import Fraction as Q

import numpy as np

from .k4_interval import DIA, IA
from .k4_path import (K4PathResult, SCOPE, _f_hi, _f_lo, _point_boxes, k4_input_digest, spans_tile_unit_interval,
                      torque_enclosure)
from .k4_reference import bernstein, dpoly, eval_poly, split


def geometry_digest(spans) -> str:
    h = hashlib.sha256()
    for sp in spans:
        h.update(f"{sp.left}|{sp.right}|".encode())
        for p in sp.q:
            h.update(("," .join(str(c) for c in p) + ";").encode())
    return h.hexdigest()


def _sub(coeffs_b, path):
    b = coeffs_b
    for ch in path:
        b = split(b)[0 if ch == "L" else 1]
    return b


def _leaf_bounds(path):
    a, b = Q(0), Q(1)
    for ch in path:
        m = (a + b) / 2
        a, b = (a, m) if ch == "L" else (m, b)
    return a, b


class GeometryCache:
    def __init__(self, model, spans, mode="box"):
        if mode not in ("box", "tm1"):
            raise ValueError("mode")
        self.model, self.mode, self.dof = model, mode, model.dof
        self.digest = geometry_digest(spans)
        self.contract_sha256 = model.contract_sha256
        self.lefts = [(sp.left, sp.right) for sp in spans]
        self.q = [[tuple(p) for p in sp.q] for sp in spans]
        self.qb = []
        for sp in spans:
            h = sp.right - sp.left
            polys = ([tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q]
                     + [dpoly(p, 3, h) for p in sp.q])
            self.qb.append((polys, [bernstein(p) for p in polys]))
        self.store = {}
        self.rnea_boxes = 0
        self.prep_seconds = 0.0

    def check(self, spans):
        if geometry_digest(spans) != self.digest or [(s.left, s.right) for s in spans] != self.lefts:
            raise ValueError("GEOMETRY_MISMATCH: cached coefficients belong to a different path")

    def ensure(self, keys):
        miss = [k for k in keys if k not in self.store]
        if not miss:
            return
        t0 = time.perf_counter()
        d = self.dof
        n = len(miss)
        # Bernstein boxes of q, q', q'', q''' on each leaf
        lo = np.empty((4 * d, n))
        hi = np.empty((4 * d, n))
        cen = np.empty((3 * d, 2, n))
        for i, (si, path) in enumerate(miss):
            polys, bern = self.qb[si]
            a, b = _leaf_bounds(path)
            for r in range(4 * d):
                sb = _sub(bern[r], path)
                lo[r, i], hi[r, i] = _f_lo(min(sb)), _f_hi(max(sb))
            if self.mode == "tm1":
                c = (a + b) / 2
                for r in range(3 * d):
                    v = eval_poly(polys[r], c)
                    cen[r, 0, i], cen[r, 1, i] = _f_lo(v), _f_hi(v)
        box = [IA(lo[r], hi[r]) for r in range(4 * d)]
        q, qp, qpp, qppp = box[:d], box[d:2 * d], box[2 * d:3 * d], box[3 * d:]
        M = self.model
        if self.mode == "box":
            A, B, G = M.coefficients(q, qp, qpp)
            for i, k in enumerate(miss):
                self.store[k] = tuple([c[j][i] for j in range(d)] for c in (A, B, G))
        else:
            cq = [IA(cen[r, 0], cen[r, 1]) for r in range(3 * d)]
            A0, B0, G0 = M.coefficients(cq[:d], cq[d:2 * d], cq[2 * d:])
            dq = [DIA(q[j], qp[j]) for j in range(d)]
            dqp = [DIA(qp[j], qpp[j]) for j in range(d)]
            dqpp = [DIA(qpp[j], qppp[j]) for j in range(d)]
            A1, B1, G1 = M.coefficients(dq, dqp, dqpp)
            for i, (si, path) in enumerate(miss):
                a, b = _leaf_bounds(path)
                H = (self.lefts[si][1] - self.lefts[si][0]) * (b - a) / 2
                self.store[(si, path)] = (
                    [A0[j][i] for j in range(d)], [B0[j][i] for j in range(d)], [G0[j][i] for j in range(d)],
                    [A1[j].d[i] for j in range(d)], [B1[j].d[i] for j in range(d)], [G1[j].d[i] for j in range(d)], H)
        self.rnea_boxes += n
        self.prep_seconds += time.perf_counter() - t0

    def certify(self, spans, lower, upper, *, max_depth=12, max_nodes=65536) -> K4PathResult:
        """Certify a timing profile (spans with the SAME geometry, new x) using cached coefficients."""
        lower = np.asarray(lower, float)
        upper = np.asarray(upper, float)
        if (lower.shape != (self.dof,) or upper.shape != (self.dof,) or not np.isfinite(lower).all()
                or not np.isfinite(upper).all() or not (lower < upper).all() or not spans_tile_unit_interval(spans)):
            return K4PathResult("INVALID_INPUT", SCOPE, 0, 0, reason="limits finite/ordered; spans tile [0, 1]")
        res = self._certify(spans, lower, upper, max_depth=max_depth, max_nodes=max_nodes)
        res.input_digest = k4_input_digest(spans, self.model.contract_sha256, lower, upper)
        return res

    def _certify(self, spans, lower, upper, *, max_depth, max_nodes):
        t0 = time.perf_counter()
        prep0 = self.prep_seconds
        self.check(spans)
        d = self.dof
        lower = np.asarray(lower, float)
        upper = np.asarray(upper, float)
        xb = []
        raws = []
        for sp in spans:
            h = sp.right - sp.left
            xp = dpoly(sp.x, 1, h)
            xb.append((bernstein(tuple(sp.x)), bernstein(xp)))
            raws.append([tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q]
                        + [tuple(sp.x), xp])
        active = [(si, "") for si in range(len(spans))]
        leaves, nodes, maxd = [], 0, 0
        worst = np.zeros(d)
        while active:
            if nodes + len(active) > max_nodes:
                return K4PathResult("UNKNOWN_NODES", SCOPE, nodes, maxd, leaves, None, worst.tolist(),
                                    time.perf_counter() - t0, "node budget")
            nodes += len(active)
            self.ensure(active)
            n = len(active)
            # x, x' Bernstein data per leaf
            xl = np.empty(n); xh = np.empty(n); pl = np.empty(n); ph = np.empty(n)
            xcoef = []
            for i, (si, path) in enumerate(active):
                bx = _sub(xb[si][0], path)
                bp = _sub(xb[si][1], path)
                xl[i], xh[i] = _f_lo(min(bx)), _f_hi(max(bx))
                pl[i], ph[i] = _f_lo(min(bp)), _f_hi(max(bp))
                if self.mode == "tm1":
                    # common degree-3 Bernstein basis for x and x' (exact degree elevation)
                    bxx, e = _elevate(list(bx), 3), _elevate(list(bp), 3)
                    xcoef.append((bxx, e))
            X, XP = IA(xl, xh), IA(pl, ph)
            entries = [self.store[k] for k in active]
            tlo = np.empty((n, d)); thi = np.empty((n, d))
            for j in range(d):
                if self.mode == "box":
                    A = _stack([e[0][j] for e in entries]); B = _stack([e[1][j] for e in entries]); G = _stack([e[2][j] for e in entries])
                    t = G + A * X + B * XP
                    tlo[:, j], thi[:, j] = t.lo, t.hi
                else:
                    A0 = _stack([e[0][j] for e in entries]); B0 = _stack([e[1][j] for e in entries]); G0 = _stack([e[2][j] for e in entries])
                    A1 = _stack([e[3][j] for e in entries]); B1 = _stack([e[4][j] for e in entries]); G1 = _stack([e[5][j] for e in entries])
                    H = np.array([float(e[6]) for e in entries])
                    Hb = IA(-np.nextafter(H, np.inf), np.nextafter(H, np.inf))
                    rem = (G1 + A1 * X + B1 * XP) * Hb
                    plo = np.full(n, np.inf); phi = np.full(n, -np.inf)
                    for k in range(4):
                        bk = IA(np.array([_f_lo(c[0][k]) for c in xcoef]), np.array([_f_hi(c[0][k]) for c in xcoef]))
                        ek = IA(np.array([_f_lo(c[1][k]) for c in xcoef]), np.array([_f_hi(c[1][k]) for c in xcoef]))
                        ck = G0 + A0 * bk + B0 * ek
                        plo = np.minimum(plo, ck.lo); phi = np.maximum(phi, ck.hi)
                    t = IA(plo, phi) + rem
                    tlo[:, j], thi[:, j] = t.lo, t.hi
            ok = (np.isfinite(tlo).all(1) & np.isfinite(thi).all(1) & (xl >= 0)
                  & (tlo >= lower).all(1) & (thi <= upper).all(1))
            util = np.maximum(np.where(thi > 0, thi / upper, 0), np.where(tlo < 0, tlo / lower, 0))
            nxt = []
            fail = [i for i in range(n) if not ok[i]]
            for i in np.nonzero(ok)[0]:
                worst = np.maximum(worst, util[i])
                si, path = active[i]
                maxd = max(maxd, len(path))
                leaves.append((si, path, tlo[i].tolist(), thi[i].tolist()))
            if fail:
                pts, rl, meta = [], [], []
                for i in fail:
                    si, path = active[i]
                    a, b = _leaf_bounds(path)
                    for u in (a, (a + b) / 2, b):
                        pts.append(u); rl.append(raws[si]); meta.append((si, u))
                pq, pqp, pqpp, px, pxp, exact = _point_boxes(rl, pts, d)
                for k, (si, u) in enumerate(meta):
                    if exact[k][3 * d] < 0:
                        return K4PathResult("VIOLATED_X", SCOPE, nodes, maxd, leaves,
                                            {"span": si, "u": str(u)}, worst.tolist(), time.perf_counter() - t0)
                ptau = torque_enclosure(self.model, pq, pqp, pqpp, px, pxp)
                plo_ = np.stack([t.lo for t in ptau], 1); phi_ = np.stack([t.hi for t in ptau], 1)
                for k, (si, u) in enumerate(meta):
                    for j in range(d):
                        if plo_[k, j] > upper[j] or phi_[k, j] < lower[j]:
                            s = spans[si].left + (spans[si].right - spans[si].left) * u
                            return K4PathResult("VIOLATED_TORQUE", SCOPE, nodes, maxd, leaves, {
                                "span": si, "u": str(u), "s": str(s), "s_float": float(s), "joint": j,
                                "torque_enclosure": [float(plo_[k, j]), float(phi_[k, j])],
                                "limit": [float(lower[j]), float(upper[j])]}, worst.tolist(),
                                time.perf_counter() - t0, "rigorous point witness")
                for i in fail:
                    si, path = active[i]
                    if len(path) >= max_depth:
                        return K4PathResult("UNKNOWN_DEPTH", SCOPE, nodes, max(maxd, len(path)), leaves, None,
                                            worst.tolist(), time.perf_counter() - t0, "subdivision exhausted")
                    nxt += [(si, path + "L"), (si, path + "R")]
            active = nxt
        res = K4PathResult("CERTIFIED", SCOPE, nodes, maxd, leaves, None, worst.tolist(),
                           time.perf_counter() - t0, f"cached {self.mode} coefficients")
        res.reason += f"; prep_in_call_s={self.prep_seconds - prep0:.4f}"
        return res


def _stack(xs):
    return IA(np.array([float(np.asarray(x.lo)) for x in xs]), np.array([float(np.asarray(x.hi)) for x in xs]))


def _elevate(b, n):
    b = list(b)
    while len(b) - 1 < n:
        m = len(b) - 1
        b = [b[0]] + [Q(k, m + 1) * b[k - 1] + (1 - Q(k, m + 1)) * b[k] for k in range(1, m + 1)] + [b[-1]]
    return b
