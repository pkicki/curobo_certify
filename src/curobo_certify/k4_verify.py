"""Independent MODEL_K4 certificate verifier (different dynamics formulation from the producer).

Checks, for a claimed CERTIFIED K4 path result:
  1. model contract hash and that the limits are the ORIGINAL physical effort limits of the contract
     (or an explicitly requested scaled contract), not producer-supplied numbers;
  2. per span, the claimed leaves form an exact dyadic partition of [0, 1] (prefix-free, measure 1);
  3. per leaf, path boxes are rebuilt by exact affine substitution of the span polynomials onto the leaf
     (not the producer's de Casteljau), and the torque enclosure is recomputed with a WORLD-FRAME
     Newton-Euler / Jacobian-transpose formulation: world rotations and points by forward kinematics,
     world inertias R Ic R^T, explicit sums tau_j = z_j . sum_{k in subtree(j)} [(c_k - o_j) x F_k + N_k].
     The producer uses link-frame recursive force propagation. Both share only the outward interval
     primitive layer (k4_interval, separately tested against mpmath) and the exact K3 span extraction.
  4. If the verifier's own enclosure on a claimed leaf is not inside the limits, it may refine that leaf
     itself (up to `refine` extra levels); any remaining failure rejects the certificate.
"""
from __future__ import annotations

from fractions import Fraction as Q

import numpy as np

from .k4_interval import IA, cross, icos, isin
from .k4_reference import bernstein, compose_subinterval, dpoly


def _hull_ia(coeff_lists):
    def lo(v):
        f = float(v)
        return f if Q(f) <= v else float(np.nextafter(f, -np.inf))

    def hi(v):
        f = float(v)
        return f if Q(f) >= v else float(np.nextafter(f, np.inf))
    k = len(coeff_lists[0])
    L = np.array([[lo(min(c[r])) for c in coeff_lists] for r in range(k)])
    H = np.array([[hi(max(c[r])) for c in coeff_lists] for r in range(k)])
    return [IA(L[r], H[r]) for r in range(k)]


class WorldFrameDynamics:
    """Independent world-frame implementation reading the same frozen contract parameters."""

    def __init__(self, contract):
        """Reads contract v2 (exact URDF decimals) with its own parser; rotations from rpy built here."""
        m = contract["model"]
        if "links" not in m:
            raise ValueError("verifier requires a v2 (rigid, URDF-exact) model contract")
        L = m["links"]
        self.n = len(L)
        D = lambda v: IA.const(Q(v))  # noqa: E731
        self.parent = [max(0, int(e["parent"])) for e in L]
        self.joint = [int(e["joint_index"]) for e in L]
        self.p_off = [[D(v) for v in e["origin_xyz"]] for e in L]
        self.Rfix = []
        for e in L:
            r, p_, y = (D(v) for v in e["origin_rpy"])
            # elementary rotations multiplied explicitly: Rz(y) @ Ry(p) @ Rx(r)
            Rx = [[D(1), D(0), D(0)], [D(0), icos(r), -isin(r)], [D(0), isin(r), icos(r)]]
            Ry = [[icos(p_), D(0), isin(p_)], [D(0), D(1), D(0)], [-isin(p_), D(0), icos(p_)]]
            Rz = [[icos(y), -isin(y), D(0)], [isin(y), icos(y), D(0)], [D(0), D(0), D(1)]]
            mm = lambda A, B: [[A[i][0] * B[0][k] + A[i][1] * B[1][k] + A[i][2] * B[2][k] for k in range(3)] for i in range(3)]  # noqa: E731
            self.Rfix.append(mm(mm(Rz, Ry), Rx))
        self.m = [D(e["mass"]) for e in L]
        self.c = [[D(v) for v in e["com"]] for e in L]
        self.I = []
        for e in L:
            x = [D(v) for v in e["inertia"]]
            self.I.append([[x[0], x[3], x[4]], [x[3], x[1], x[5]], [x[4], x[5], x[2]]])
        self.g = [D(v) for v in m["gravity"]]
        self.dof = int(m["num_dof"])
        self.sub = [set() for _ in range(self.n)]
        for k in range(self.n):
            i = k
            while True:
                self.sub[i].add(k)
                if i == 0:
                    break
                i = self.parent[i]

    def torque(self, q, v, a, gravity):
        shape = q[0].shape
        Z = IA(np.zeros(shape))
        one = IA(np.ones(shape))
        Rw = [None] * self.n
        o = [None] * self.n
        w = [None] * self.n
        wd = [None] * self.n
        ao = [None] * self.n
        Rw[0] = [[one, Z, Z], [Z, one, Z], [Z, Z, one]]  # Rw[r][c]
        o[0] = [Z, Z, Z]
        w[0], wd[0], ao[0] = [Z, Z, Z], [Z, Z, Z], [Z, Z, Z]
        zax = [None] * self.n
        for i in range(1, self.n):
            p = self.parent[i]
            Rf, pf = self.Rfix[i], self.p_off[i]
            j = self.joint[i]
            if j >= 0:
                cq, sq = icos(q[j]), isin(q[j])
                Rl = [[cq * Rf[r][0] + sq * Rf[r][1], cq * Rf[r][1] - sq * Rf[r][0], Rf[r][2] + Z] for r in range(3)]
            else:
                Rl = [[Rf[r][k] + Z for k in range(3)] for r in range(3)]
            Rp = Rw[p]
            Rw[i] = [[Rp[r][0] * Rl[0][k] + Rp[r][1] * Rl[1][k] + Rp[r][2] * Rl[2][k] for k in range(3)] for r in range(3)]
            dp = [Rp[r][0] * pf[0] + Rp[r][1] * pf[1] + Rp[r][2] * pf[2] for r in range(3)]
            o[i] = [o[p][r] + dp[r] for r in range(3)]
            # point acceleration of o_i (rigidly attached to parent)
            t = cross(wd[p], dp)
            ao[i] = [ao[p][r] + t[r] for r in range(3)]
            if v is not None:
                t = cross(w[p], cross(w[p], dp))
                ao[i] = [ao[i][r] + t[r] for r in range(3)]
            if j >= 0:
                z = [Rw[i][r][2] for r in range(3)]
                zax[i] = z
                qd = Z if v is None else v[j]
                qdd = Z if a is None else a[j]
                w[i] = [w[p][r] + z[r] * qd for r in range(3)] if v is not None else [Z, Z, Z]
                wd[i] = [wd[p][r] + z[r] * qdd for r in range(3)]
                if v is not None:
                    t = cross(w[p], [z[r] * qd for r in range(3)])
                    wd[i] = [wd[i][r] + t[r] for r in range(3)]
            else:
                w[i], wd[i] = w[p], wd[p]
        # body wrenches (world frame) about each CoM
        cw, F, N = [None] * self.n, [None] * self.n, [None] * self.n
        gvec = [x + Z for x in self.g]
        for k in range(1, self.n):
            R = Rw[k]
            rc = [R[r][0] * self.c[k][0] + R[r][1] * self.c[k][1] + R[r][2] * self.c[k][2] for r in range(3)]
            cw[k] = [o[k][r] + rc[r] for r in range(3)]
            acc = [ao[k][r] for r in range(3)]
            t = cross(wd[k], rc)
            acc = [acc[r] + t[r] for r in range(3)]
            if v is not None:
                t = cross(w[k], cross(w[k], rc))
                acc = [acc[r] + t[r] for r in range(3)]
            if gravity:
                acc = [acc[r] - gvec[r] for r in range(3)]
            F[k] = [x * self.m[k] for x in acc]
            # world inertia applied: I_w u = R (Ic (R^T u))
            def Iw(u, R=R, Ic=self.I[k]):
                bu = [R[0][c] * u[0] + R[1][c] * u[1] + R[2][c] * u[2] for c in range(3)]
                ib = [bu[0] * Ic[r][0] + bu[1] * Ic[r][1] + bu[2] * Ic[r][2] for r in range(3)]
                return [R[r][0] * ib[0] + R[r][1] * ib[1] + R[r][2] * ib[2] for r in range(3)]
            N[k] = Iw(wd[k])
            if v is not None:
                t = cross(w[k], Iw(w[k]))
                N[k] = [N[k][r] + t[r] for r in range(3)]
        tau = [None] * self.dof
        for i in range(1, self.n):
            j = self.joint[i]
            if j < 0:
                continue
            acc = [Z, Z, Z]
            for k in self.sub[i]:
                if k == 0:
                    continue
                arm = [cw[k][r] - o[i][r] for r in range(3)]
                t = cross(arm, F[k])
                acc = [acc[r] + t[r] + N[k][r] for r in range(3)]
            z = zax[i]
            tau[j] = z[0] * acc[0] + z[1] * acc[1] + z[2] * acc[2]
        return tau

    def leaf_torque(self, q, qp, qpp, x, xp):
        G = self.torque(q, None, None, True)
        B = [b.scale(0.5) for b in self.torque(q, None, qp, False)]
        A = self.torque(q, qp, qpp, False)
        return [G[j] + A[j] * x + B[j] * xp for j in range(self.dof)]


def _leaf_interval(path: str):
    a, b = Q(0), Q(1)
    for ch in path:
        m = (a + b) / 2
        if ch == "L":
            b = m
        elif ch == "R":
            a = m
        else:
            raise ValueError("bad path symbol")
    return a, b


FROZEN_CONTRACT_V2_SHA256 = "9caa38bff95b56df89ea6dc71a08a5558474e764a613ecc480c4c38364f663e7"


def _contract_ok(contract, expected):
    import hashlib
    import json
    body = json.dumps(contract["model"], sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256(body).hexdigest()
    return h == contract.get("contract_sha256") and h == expected


def verify_k4(spans, contract, result, lower, upper, *, expect_physical=True, refine=6, batch=4096,
              expected_contract_sha256=FROZEN_CONTRACT_V2_SHA256):
    """Return dict(pass, reason, leaves_checked, refined_leaves). Never raises on malformed input."""
    try:
        return _verify_k4(spans, contract, result, lower, upper, expect_physical=expect_physical, refine=refine,
                          batch=batch, expected=expected_contract_sha256)
    except (ValueError, KeyError, TypeError, ZeroDivisionError, AttributeError, IndexError) as ex:
        return {"pass": False, "reason": f"malformed certificate: {type(ex).__name__}: {ex}"}


def _verify_k4(spans, contract, result, lower, upper, *, expect_physical, refine, batch, expected):
    from .k4_path import k4_input_digest, spans_tile_unit_interval
    if not _contract_ok(contract, expected):
        return {"pass": False, "reason": "model contract hash mismatch (recomputed != recorded/frozen)"}
    if not spans_tile_unit_interval(spans):
        return {"pass": False, "reason": "spans do not tile [0, 1]"}
    lo_, hi_ = np.asarray(lower, float), np.asarray(upper, float)
    if not (np.isfinite(lo_).all() and np.isfinite(hi_).all()):
        return {"pass": False, "reason": "non-finite limits"}
    m = contract["model"]
    phys_lo = np.array([float(Q(v)) for v in m["effort_lower"]])   # exact integers (87 / 12)
    phys_hi = np.array([float(Q(v)) for v in m["effort_upper"]])
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if expect_physical and not (np.array_equal(lower, phys_lo) and np.array_equal(upper, phys_hi)):
        return {"pass": False, "reason": "limits are not the contract's physical effort limits"}
    if result.status != "CERTIFIED":
        return {"pass": False, "reason": "not a certificate"}
    if not result.input_digest or result.input_digest != k4_input_digest(spans, contract["contract_sha256"], lower, upper):
        return {"pass": False, "reason": "result input digest missing or does not bind these spans/contract/limits"}
    for (_si, _p, blo, bhi) in result.leaves:  # recorded producer boxes must themselves be inside the limits
        if not (np.isfinite(blo).all() and np.isfinite(bhi).all() and (np.asarray(blo) >= lower).all()
                and (np.asarray(bhi) <= upper).all()):
            return {"pass": False, "reason": "recorded leaf box outside limits"}
    W = WorldFrameDynamics(contract)
    dof = W.dof
    # Recorded boxes must actually enclose the torque: rigorous world-frame point torque at each leaf
    # midpoint must lie inside the recorded box (rejects zeroed / too-narrow boxes; review F-P2-4).
    from .k4_reference import eval_poly
    mids, rec = [], []
    if any(not (isinstance(si, int) and 0 <= si < len(spans)) for (si, *_r) in result.leaves):
        return {"pass": False, "reason": "leaf refers to a span outside the path"}
    for (si, path, blo, bhi) in result.leaves:
        a_, b_ = _leaf_interval(path)
        sp = spans[si]
        h = sp.right - sp.left
        u = (a_ + b_) / 2
        vals = [eval_poly(p, u) for p in ([tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q]
                                         + [dpoly(p, 2, h) for p in sp.q] + [tuple(sp.x), dpoly(sp.x, 1, h)])]
        mids.append(vals)
        rec.append((blo, bhi))
    if mids:
        cols = list(zip(*mids))
        bx = [IA(np.array([float(np.nextafter(float(v), -np.inf)) for v in col]),
                 np.array([float(np.nextafter(float(v), np.inf)) for v in col])) for col in cols]
        pt = W.leaf_torque(bx[:dof], bx[dof:2 * dof], bx[2 * dof:3 * dof], bx[3 * dof], bx[3 * dof + 1])
        plo = np.stack([t.lo for t in pt], 1)
        phi = np.stack([t.hi for t in pt], 1)
        RL = np.array([r[0] for r in rec]); RH = np.array([r[1] for r in rec])
        if not ((plo >= RL - 1e-9 * np.maximum(1, np.abs(RL))) & (phi <= RH + 1e-9 * np.maximum(1, np.abs(RH)))).all():
            return {"pass": False, "reason": "recorded leaf box does not contain the rigorous midpoint torque"}
    by_span = {}
    for (si, path, _lo, _hi) in result.leaves:
        by_span.setdefault(si, []).append(path)
    if set(by_span) != set(range(len(spans))):
        return {"pass": False, "reason": "missing span"}
    for si, paths in by_span.items():
        if len(set(paths)) != len(paths) or sum(Q(1, 2 ** len(p)) for p in paths) != 1:
            return {"pass": False, "reason": f"span {si}: not a partition"}
        ps = sorted(paths)
        if any(ps[i + 1].startswith(ps[i]) for i in range(len(ps) - 1)):
            return {"pass": False, "reason": f"span {si}: overlapping leaves"}
    work = []
    for si, paths in by_span.items():
        sp = spans[si]
        h = sp.right - sp.left
        raw = ([tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q]
               + [tuple(sp.x), dpoly(sp.x, 1, h)])
        for p in paths:
            work.append((si, raw, *_leaf_interval(p), 0))
    checked, refined = 0, 0
    while work:
        nxt = []
        for c0 in range(0, len(work), batch):
            chunk = work[c0:c0 + batch]
            coeffs = [tuple(bernstein(compose_subinterval(p, a, b)) for p in raw) for (_si, raw, a, b, _d) in chunk]
            boxes = _hull_ia(coeffs)
            q, qp, qpp, x, xp = boxes[:dof], boxes[dof:2 * dof], boxes[2 * dof:3 * dof], boxes[3 * dof], boxes[3 * dof + 1]
            tau = W.leaf_torque(q, qp, qpp, x, xp)
            tlo = np.stack([t.lo for t in tau], 1)
            thi = np.stack([t.hi for t in tau], 1)
            ok = (np.isfinite(tlo).all(1) & np.isfinite(thi).all(1) & (x.lo >= 0)
                  & (tlo >= lower).all(1) & (thi <= upper).all(1))
            checked += len(chunk)
            for i in np.nonzero(~ok)[0]:
                si, raw, a, b, d = chunk[i]
                if d >= refine:
                    return {"pass": False, "reason": f"span {si} leaf [{a},{b}] not verified after {refine} refinements",
                            "leaves_checked": checked, "refined_leaves": refined}
                mid = (a + b) / 2
                refined += 1
                nxt += [(si, raw, a, mid, d + 1), (si, raw, mid, b, d + 1)]
        work = nxt
    return {"pass": True, "reason": "partition complete; world-frame enclosures inside physical limits",
            "leaves_checked": checked, "refined_leaves": refined}


def verify_k4_witness(spans, contract, result, lower, upper, *, expected_contract_sha256=FROZEN_CONTRACT_V2_SHA256) -> dict:
    """Independent replay of a VIOLATED_TORQUE claim: exact rational point s on the claimed span,
    world-frame point torque enclosure, strict disjointness from the claimed joint's limit."""
    from .k4_reference import eval_poly
    w = getattr(result, "witness", None)
    if getattr(result, "status", None) != "VIOLATED_TORQUE" or not isinstance(w, dict):
        return {"pass": False, "reason": "not a torque violation claim"}
    try:
        from .k4_path import k4_input_digest
        if not _contract_ok(contract, expected_contract_sha256):
            return {"pass": False, "reason": "model contract hash mismatch"}
        if not result.input_digest or result.input_digest != k4_input_digest(spans, contract["contract_sha256"], lower, upper):
            return {"pass": False, "reason": "result input digest missing or mismatched"}
        W = WorldFrameDynamics(contract)
        si, j, u = int(w["span"]), int(w["joint"]), Q(w["u"])
        if not (0 <= si < len(spans) and 0 <= j < W.dof and 0 <= u <= 1):
            return {"pass": False, "reason": "witness out of range"}
        sp = spans[si]
        h = sp.right - sp.left
        raw = ([tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q]
               + [tuple(sp.x), dpoly(sp.x, 1, h)])
        vals = [eval_poly(p, u) for p in raw]
        if vals[3 * W.dof] < 0:
            return {"pass": False, "reason": "x < 0 at witness (VIOLATED_X, not torque)"}
        box = [IA.const(v) for v in vals]
        d = W.dof
        tau = W.leaf_torque(box[:d], box[d:2 * d], box[2 * d:3 * d], box[3 * d], box[3 * d + 1])[j]
        lo, hi = float(np.asarray(tau.lo)), float(np.asarray(tau.hi))
        ok = lo > float(upper[j]) or hi < float(lower[j])
        return {"pass": bool(ok), "reason": "point torque enclosure disjoint from limit" if ok else "not disjoint",
                "enclosure": [lo, hi], "s": str(sp.left + h * u)}
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as ex:
        return {"pass": False, "reason": f"malformed witness: {ex}"}
