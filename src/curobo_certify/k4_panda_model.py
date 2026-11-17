"""K4 Panda dynamics: frozen cuRobo model contract and rigorous interval RNEA.

MODEL (scope CUROBO_PINNED_PANDA_RIGID_BODY_MODEL): the rigid-body tree exactly as stored by
pinned cuRobo c301cb00 TimeOpt (franka.yml) - 13 links, float32 parameters interpreted as
exact reals: parent map, fixed transforms [R|p] (child joint frame in parent frame), Z-axis
revolute joints, masses, CoM and CoM-frame inertia tensors, gravity. Fixed links are kept as
exact rigid attachments (cuRobo lumps them numerically; mathematically identical).
This is cuRobo's CONSTRAINT model; its physical fidelity is NOT established (see contract
`physical_validity_findings`).

Interval RNEA: classic link-frame recursive Newton-Euler in outward binary64 interval
arithmetic (k4_interval). For path boxes q, q', q'' it returns enclosures of
    G = g(q)                      (RNEA with gravity, zero velocity/acceleration)
    B = 1/2 M(q) q'               (1/2 x RNEA without gravity, v = 0, a = q')
    A = M(q) q'' + C(q, q') q'    (RNEA without gravity, v = q', a = q'')
without subtracting enclosures, so tau(s) = A x + B x' + G (cuRobo TimeOpt's own identity).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fractions import Fraction

import numpy as np

from .k4_interval import IA, cross, icos, isin

SCOPE = "CUROBO_PINNED_PANDA_RIGID_BODY_MODEL"
Z_ROT, FIXED = 5, -1


def f32_hex(a) -> list:
    return [float(x).hex() for x in np.asarray(a, dtype=np.float32).astype(np.float64).reshape(-1)]


def _dec(v) -> IA:
    """Exact decimal string / Fraction -> outward scalar interval (exact if representable)."""
    return IA.const(Fraction(v))


def _rpy(r, p, y):
    """Interval enclosure of R = Rz(y) Ry(p) Rx(r) for exact decimal angles (URDF convention)."""
    cr, sr = icos(_dec(r)), isin(_dec(r))
    cp, sp = icos(_dec(p)), isin(_dec(p))
    cy, sy = icos(_dec(y)), isin(_dec(y))
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


class PandaModel:
    """Frozen rigid-body model. Accepts contract v2 (exact URDF decimals, rpy rotations) or a
    hex-float parameterisation (v1 / test fixtures). All parameters are held as scalar intervals."""

    def __init__(self, contract: dict, *, payload_override: dict | None = None):
        if payload_override is not None:
            raise ValueError("payload overrides are BLOCKED_PAYLOAD: no sourced payload model")
        m = contract["model"]
        self.contract_sha256 = contract["contract_sha256"]
        if "links" in m:
            L = m["links"]
            self.n = len(L)
            self.parent = [max(0, int(e["parent"])) for e in L]
            self.joint = [int(e["joint_index"]) for e in L]
            self.jtype = [Z_ROT if e["joint_type"] == "Z_ROT" else FIXED for e in L]
            self.Rf = [_rpy(*e["origin_rpy"]) for e in L]
            self.pf = [[_dec(v) for v in e["origin_xyz"]] for e in L]
            self.mass = [_dec(e["mass"]) for e in L]
            self.com = [[_dec(v) for v in e["com"]] for e in L]
            Ic = [[_dec(v) for v in e["inertia"]] for e in L]
            self.gravity = [_dec(v) for v in m["gravity"]]
        else:
            dec = lambda key, shape: np.array([float.fromhex(h) for h in m[key]], dtype=np.float64).reshape(shape)  # noqa: E731
            self.n = int(m["num_links"])
            self.parent = [int(x) for x in m["parent"]]
            self.joint = [int(x) for x in m["joint_index"]]
            self.jtype = [int(x) for x in m["joint_type"]]
            T = dec("fixed_transforms_hex", (self.n, 3, 4))
            self.Rf = [[[IA(T[i][r, k]) for k in range(3)] for r in range(3)] for i in range(self.n)]
            self.pf = [[IA(T[i][r, 3]) for r in range(3)] for i in range(self.n)]
            mc = dec("masses_com_hex", (self.n, 4))
            self.mass = [IA(mc[i, 3]) for i in range(self.n)]
            self.com = [[IA(mc[i, k]) for k in range(3)] for i in range(self.n)]
            ine = dec("inertias_hex", (self.n, 6))
            Ic = [[IA(v) for v in ine[i]] for i in range(self.n)]
            self.gravity = [IA(float.fromhex(h)) for h in m["gravity_hex"]]
        # CoM inertia matrices from (ixx iyy izz ixy ixz iyz)
        self.Ic = [[[r[0], r[3], r[4]], [r[3], r[1], r[5]], [r[4], r[5], r[2]]] for r in Ic]
        self.dof = int(m["num_dof"])
        for i in range(1, self.n):
            if not (0 <= self.parent[i] < i):
                raise ValueError("links must be topologically ordered")
            if self.jtype[i] not in (Z_ROT, FIXED) or (self.jtype[i] == Z_ROT) != (self.joint[i] >= 0):
                raise ValueError("unsupported joint type in contract")
        if sorted(j for j in self.joint if j >= 0) != list(range(self.dof)):
            raise ValueError("joint map must cover 0..dof-1 once")

    # ------------------------------------------------------------------------------------
    def rnea(self, q, v=None, a=None, *, gravity: bool):
        """Interval inverse dynamics. q, v, a: lists (len dof) of IA with equal shapes
        (v / a = None means exactly zero). Returns list of dof IA torques."""
        shape = q[0].shape
        Z = IA(np.zeros(shape))
        g0 = [-g for g in self.gravity] if gravity else [Z, Z, Z]
        g0 = [x + Z for x in g0]
        w = [None] * self.n
        wd = [None] * self.n
        acc = [None] * self.n
        R = [None] * self.n
        w[0], wd[0], acc[0] = [Z, Z, Z], [Z, Z, Z], g0
        zero_v = v is None
        for i in range(1, self.n):
            p = self.parent[i]
            Rf, pf = self.Rf[i], self.pf[i]
            if self.jtype[i] == Z_ROT:
                j = self.joint[i]
                c, s = icos(q[j]), isin(q[j])
                col0 = [c * Rf[r][0] + s * Rf[r][1] for r in range(3)]
                col1 = [c * Rf[r][1] - s * Rf[r][0] for r in range(3)]
                col2 = [Rf[r][2] + Z for r in range(3)]
            else:
                j = -1
                col0, col1, col2 = ([Rf[r][k] + Z for r in range(3)] for k in range(3))
            Ri = [col0, col1, col2]  # Ri[k][r] = R[r, k]
            R[i] = Ri

            def rt(vec, Ri=Ri):  # R^T vec
                return [Ri[k][0] * vec[0] + Ri[k][1] * vec[1] + Ri[k][2] * vec[2] for k in range(3)]
            wp, wdp, ap = w[p], wd[p], acc[p]
            t1 = cross(wdp, pf)
            a_par = [ap[k] + t1[k] for k in range(3)]
            if not zero_v:
                t2 = cross(wp, cross(wp, pf))
                a_par = [a_par[k] + t2[k] for k in range(3)]
            acc[i] = rt(a_par)
            wr = rt(wp) if not zero_v else [Z, Z, Z]
            wdr = rt(wdp)
            if j >= 0:
                qd = Z if zero_v else v[j]
                qdd = Z if a is None else a[j]
                w[i] = [wr[0], wr[1], wr[2] + qd] if not zero_v else [Z, Z, Z]
                wd[i] = [wdr[0], wdr[1], wdr[2] + qdd]
                if not zero_v:
                    wd[i] = [wd[i][0] + wr[1] * qd, wd[i][1] - wr[0] * qd, wd[i][2]]
            else:
                w[i], wd[i] = wr, wdr
        f = [[Z, Z, Z] for _ in range(self.n)]
        nmom = [[Z, Z, Z] for _ in range(self.n)]
        for i in range(1, self.n):
            m, c, Ic = self.mass[i], self.com[i], self.Ic[i]
            t = cross(wd[i], c)
            ac = [acc[i][k] + t[k] for k in range(3)]
            if not zero_v:
                t = cross(w[i], cross(w[i], c))
                ac = [ac[k] + t[k] for k in range(3)]
            F = [x * m for x in ac]
            N = _matvec(Ic, wd[i])
            if not zero_v:
                t = cross(w[i], _matvec(Ic, w[i]))
                N = [N[k] + t[k] for k in range(3)]
            cF = cross(c, F)
            f[i] = [f[i][k] + F[k] for k in range(3)]
            nmom[i] = [nmom[i][k] + N[k] + cF[k] for k in range(3)]
        tau = [None] * self.dof
        for i in range(self.n - 1, 0, -1):
            if self.joint[i] >= 0:
                tau[self.joint[i]] = nmom[i][2]
            p = self.parent[i]
            Ri = R[i]
            Rf_ = [Ri[0][r] * f[i][0] + Ri[1][r] * f[i][1] + Ri[2][r] * f[i][2] for r in range(3)]
            Rn = [Ri[0][r] * nmom[i][0] + Ri[1][r] * nmom[i][1] + Ri[2][r] * nmom[i][2] for r in range(3)]
            pxf = cross(self.pf[i], Rf_)
            f[p] = [f[p][k] + Rf_[k] for k in range(3)]
            nmom[p] = [nmom[p][k] + Rn[k] + pxf[k] for k in range(3)]
        return tau

    def coefficients(self, q, qp, qpp):
        """Enclosures of A, B, G (each a list of dof IA) for path boxes q, q', q''."""
        G = self.rnea(q, None, None, gravity=True)
        B = [b.scale(0.5) for b in self.rnea(q, None, qp, gravity=False)]
        A = self.rnea(q, qp, qpp, gravity=False)
        return A, B, G

    def torque(self, q, qd, qdd):
        return self.rnea(q, qd, qdd, gravity=True)


def _matvec(M, v):
    return [M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2] for r in range(3)]


def load_contract(path: Path) -> dict:
    raw = Path(path).read_bytes()
    c = json.loads(raw)
    body = json.dumps(c["model"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(body).hexdigest() != c["contract_sha256"]:
        raise ValueError("model contract hash mismatch")
    return c
