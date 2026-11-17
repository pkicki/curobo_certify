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

import numpy as np

from .k4_interval import IA, cross, icos, isin

SCOPE = "CUROBO_PINNED_PANDA_RIGID_BODY_MODEL"
Z_ROT, FIXED = 5, -1


def f32_hex(a) -> list:
    return [float(x).hex() for x in np.asarray(a, dtype=np.float32).astype(np.float64).reshape(-1)]


class PandaModel:
    """Frozen model from a contract dict (see k4_model_contract.py)."""

    def __init__(self, contract: dict, *, payload_override: dict | None = None):
        if payload_override is not None:
            raise ValueError("payload overrides are BLOCKED_PAYLOAD: no sourced payload model")
        m = contract["model"]
        self.contract_sha256 = contract["contract_sha256"]
        dec = lambda key, shape: np.array([float.fromhex(h) for h in m[key]], dtype=np.float64).reshape(shape)  # noqa: E731
        self.n = int(m["num_links"])
        self.parent = [int(x) for x in m["parent"]]
        self.joint = [int(x) for x in m["joint_index"]]
        self.jtype = [int(x) for x in m["joint_type"]]
        self.T = dec("fixed_transforms_hex", (self.n, 3, 4))
        mc = dec("masses_com_hex", (self.n, 4))
        self.com, self.mass = mc[:, :3], mc[:, 3]
        ine = dec("inertias_hex", (self.n, 6))  # ixx iyy izz ixy ixz iyz about CoM
        self.Ic = np.array([[[r[0], r[3], r[4]], [r[3], r[1], r[5]], [r[4], r[5], r[2]]] for r in ine])
        self.gravity = np.array([float.fromhex(h) for h in m["gravity_hex"]])
        self.dof = int(m["num_dof"])
        for i in range(1, self.n):
            if not (0 <= self.parent[i] < i):
                raise ValueError("links must be topologically ordered")
            if self.jtype[i] not in (Z_ROT, FIXED) or (self.jtype[i] == Z_ROT) != (self.joint[i] >= 0):
                raise ValueError("unsupported joint type in contract")
        if sorted(j for j in self.joint if j >= 0) != list(range(self.dof)):
            raise ValueError("joint map must cover 0..dof-1 once")
        self.children = [[c for c in range(1, self.n) if self.parent[c] == i] for i in range(self.n)]

    # ------------------------------------------------------------------------------------
    def rnea(self, q, v=None, a=None, *, gravity: bool):
        """Interval inverse dynamics. q, v, a: lists (len dof) of IA with equal shapes
        (v / a = None means exactly zero). Returns list of dof IA torques."""
        shape = q[0].shape
        Z = IA(np.zeros(shape))
        g0 = [IA(np.full(shape, -x)) for x in self.gravity] if gravity else [Z, Z, Z]
        w = [None] * self.n
        wd = [None] * self.n
        acc = [None] * self.n
        R = [None] * self.n
        w[0], wd[0], acc[0] = [Z, Z, Z], [Z, Z, Z], g0
        zero_v = v is None
        for i in range(1, self.n):
            p = self.parent[i]
            Rf = self.T[i][:, :3]
            pf = self.T[i][:, 3]
            if self.jtype[i] == Z_ROT:
                j = self.joint[i]
                c, s = icos(q[j]), isin(q[j])
                col0 = [c.scale(Rf[r, 0]) + s.scale(Rf[r, 1]) for r in range(3)]
                col1 = [c.scale(Rf[r, 1]) - s.scale(Rf[r, 0]) for r in range(3)]
                col2 = [IA(np.full(shape, Rf[r, 2])) for r in range(3)]
            else:
                j = -1
                col0, col1, col2 = ([IA(np.full(shape, Rf[r, k])) for r in range(3)] for k in range(3))
            Ri = [col0, col1, col2]  # Ri[k][r] = R[r, k]
            R[i] = Ri

            def rt(vec):  # R^T vec
                return [Ri[k][0] * vec[0] + Ri[k][1] * vec[1] + Ri[k][2] * vec[2] for k in range(3)]
            pfv = [float(x) for x in pf]
            wp, wdp, ap = w[p], wd[p], acc[p]
            # linear acceleration of child origin, in parent frame, then rotated
            a_par = [ap[k] for k in range(3)]
            t1 = _cross_c(wdp, pfv)
            a_par = [a_par[k] + t1[k] for k in range(3)]
            if not zero_v:
                t2 = cross(wp, _cross_c(wp, pfv))
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
        # body forces
        f = [[Z, Z, Z] for _ in range(self.n)]
        nmom = [[Z, Z, Z] for _ in range(self.n)]
        for i in range(1, self.n):
            m, c = float(self.mass[i]), [float(x) for x in self.com[i]]
            ac = [acc[i][k] for k in range(3)]
            t = _cross_c(wd[i], c)
            ac = [ac[k] + t[k] for k in range(3)]
            if not zero_v:
                t = cross(w[i], _cross_c(w[i], c))
                ac = [ac[k] + t[k] for k in range(3)]
            F = [x.scale(m) for x in ac]
            Icw = _matvec_c(self.Ic[i], wd[i])
            N = Icw
            if not zero_v:
                N = [N[k] + cross(w[i], _matvec_c(self.Ic[i], w[i]))[k] for k in range(3)]
            cF = _c_cross(c, F)
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
            pf = [float(x) for x in self.T[i][:, 3]]
            pxf = _c_cross(pf, Rf_)
            f[p] = [f[p][k] + Rf_[k] for k in range(3)]
            nmom[p] = [nmom[p][k] + Rn[k] + pxf[k] for k in range(3)]
        return tau

    def coefficients(self, q, qp, qpp):
        """Enclosures of A, B, G (each a list of dof IA) for path boxes q, q', q''."""
        G = self.rnea(q, None, None, gravity=True)
        Bfull = self.rnea(q, None, qp, gravity=False)
        B = [b.scale(0.5) for b in Bfull]
        A = self.rnea(q, qp, qpp, gravity=False)
        return A, B, G

    def torque(self, q, qd, qdd):
        return self.rnea(q, qd, qdd, gravity=True)


def _cross_c(a, c):
    """a (interval 3-vector) x c (exact float 3-vector)."""
    return [a[1].scale(c[2]) - a[2].scale(c[1]), a[2].scale(c[0]) - a[0].scale(c[2]), a[0].scale(c[1]) - a[1].scale(c[0])]


def _c_cross(c, a):
    """c (exact float 3-vector) x a (interval 3-vector)."""
    return [a[2].scale(c[1]) - a[1].scale(c[2]), a[0].scale(c[2]) - a[2].scale(c[0]), a[1].scale(c[0]) - a[0].scale(c[1])]


def _matvec_c(M, v):
    return [v[0].scale(M[r, 0]) + v[1].scale(M[r, 1]) + v[2].scale(M[r, 2]) for r in range(3)]


def load_contract(path: Path) -> dict:
    raw = Path(path).read_bytes()
    c = json.loads(raw)
    body = json.dumps(c["model"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(body).hexdigest() != c["contract_sha256"]:
        raise ValueError("model contract hash mismatch")
    return c
