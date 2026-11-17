"""Independent P2 oracle for Panda inverse dynamics (reviewer-written; imports nothing from src/).

Model: built here from the pinned URDF by the reviewer's own XML parse plus cuRobo's documented
franka.yml settings (lock_joints 0.04 m, extra link `attached_object` at panda_hand with identity
transform, cuRobo default inertial 0.01 kg / 1e-4 for links without one, links not loaded by
cuRobo excluded). The result is compared field-by-field with contract v2 in the tests.

Dynamics: ENERGY / EULER-LAGRANGE, not Newton-Euler:
  * forward kinematics R_k(q), c_k(q) by homogeneous composition (URDF semantics: child frame =
    parent * Trans(xyz) * Rpy(rpy) * Rz(q));
  * kinetic energy T(q, v) = 1/2 sum m_k |dc_k/dt|^2 + 1/2 Omega_k^T I_k Omega_k, with
    Omega_k^ = R_k^T dR_k/dt, the time derivatives obtained by forward-mode AD (dual number in the
    velocity direction v);
  * M(q) w by polarization of the quadratic form T(q, .) (exact algebra);
  * tau = M(q) qdd + Mdot(q; qd) qd - grad_q T(q, qd) + grad_q V(q), with Mdot and grad_q by a
    second, outer layer of forward-mode AD (nested duals);
  * V(q) = - sum m_k g . c_k(q).
Scalar context: mpmath.iv (rigorous outward intervals; all AD is exact algebra, so the result is
a rigorous enclosure) or mpmath.mp (fast high-precision point values for dense sampling).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

import mpmath
from mpmath import iv, mp

PINNED = Path("/home/pkicki/curobo-pinned-c301cb00a87012c2808b1a60649c4065c9b3c95a/curobo/content")
URDF = PINNED / "assets/robot/franka_description/franka_panda.urdf"
YML = PINNED / "configs/robot/franka.yml"


# ======================================================================== scalar contexts
class Ctx:
    def __init__(self, kind="iv", prec=160):
        self.kind = kind
        self.prec = prec
        self.m = iv if kind == "iv" else mp

    def setup(self):
        self.m.prec = self.prec

    def const(self, fr):
        fr = Q(fr)
        return self.m.mpf(fr.numerator) / self.m.mpf(fr.denominator)

    def sin(self, x):
        return x.sin() if isinstance(x, Dual) else self.m.sin(x)

    def cos(self, x):
        return x.cos() if isinstance(x, Dual) else self.m.cos(x)

    def sqrt(self, x):
        return self.m.sqrt(x)


_CTX = [None]


def _ad(a, b):
    if isinstance(b, Dual) and not isinstance(a, Dual):
        return b + a
    return a + b


def _sb(a, b):
    if isinstance(b, Dual) and not isinstance(a, Dual):
        return (-b) + a
    return a - b


def _ml(a, b):
    if isinstance(b, Dual) and not isinstance(a, Dual):
        return b * a
    return a * b


class Dual:
    """Forward-mode dual number val + sum_i d[i] eps_i (eps_i eps_j = 0); entries may be Duals."""
    __slots__ = ("v", "d")

    def __init__(self, v, d):
        self.v, self.d = v, d

    def __add__(self, o):
        if isinstance(o, Dual):
            return Dual(_ad(self.v, o.v), [_ad(a, b) for a, b in zip(self.d, o.d)])
        return Dual(_ad(self.v, o), self.d)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, [-a for a in self.d])

    def __sub__(self, o):
        if isinstance(o, Dual):
            return Dual(_sb(self.v, o.v), [_sb(a, b) for a, b in zip(self.d, o.d)])
        return Dual(_sb(self.v, o), self.d)

    def __rsub__(self, o):
        return Dual(_sb(o, self.v), [-a for a in self.d])

    def __mul__(self, o):
        if isinstance(o, Dual):
            return Dual(_ml(self.v, o.v), [_ad(_ml(self.v, b), _ml(a, o.v)) for a, b in zip(self.d, o.d)])
        return Dual(_ml(self.v, o), [_ml(a, o) for a in self.d])

    __rmul__ = __mul__

    def sin(self):
        C = _CTX[0]
        c = C.cos(self.v)
        return Dual(C.sin(self.v), [_ml(c, a) for a in self.d])

    def cos(self):
        C = _CTX[0]
        s = C.sin(self.v)
        return Dual(C.cos(self.v), [-_ml(s, a) for a in self.d])


def dval(x):
    return x.v if isinstance(x, Dual) else x


# ======================================================================== model from URDF
@dataclass
class OLink:
    name: str
    parent: int
    joint: int            # 0..6 or -1
    xyz: tuple            # Fractions
    rpy: tuple            # Fractions
    mass: Q
    com: tuple
    inertia: tuple        # ixx iyy izz ixy ixz iyz


def parse_urdf_model(urdf=URDF):
    """Reviewer's own URDF reading. Returns (links, gravity)."""
    root = ET.parse(urdf).getroot()
    joints = root.findall("joint")
    links = {l.get("name"): l for l in root.findall("link")}
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j.find("parent").get("link"), []).append(j)
    # cuRobo's loaded tree (franka.yml): base panda_link0, revolute chain, panda_link8, panda_hand,
    # locked fingers; ee_link and right_gripper are NOT loaded (see tests: v1 float32 masses prove it).
    loaded = {"panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5",
              "panda_link6", "panda_link7", "panda_link8", "panda_hand", "panda_leftfinger",
              "panda_rightfinger"}
    lock = {"panda_finger_joint1": Q("0.04"), "panda_finger_joint2": Q("0.04")}
    out = []
    names = []

    def inertial(name):
        l = links.get(name)
        i = None if l is None else l.find("inertial")
        if i is None:
            return Q("0.01"), (Q(0),) * 3, (Q("0.0001"),) * 3 + (Q(0),) * 3
        o = i.find("origin")
        assert [Q(v) for v in o.get("rpy").split()] == [0, 0, 0]
        ii = i.find("inertia")
        return (Q(i.find("mass").get("value")), tuple(Q(v) for v in o.get("xyz").split()),
                tuple(Q(ii.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))

    def visit(name, parent_idx, joint_el):
        idx = len(out)
        names.append(name)
        if joint_el is None:
            xyz, rpy, jidx = (Q(0),) * 3, (Q(0),) * 3, -1
        else:
            o = joint_el.find("origin")
            xyz = [Q(v) for v in (o.get("xyz") or "0 0 0").split()]
            rpy = tuple(Q(v) for v in (o.get("rpy") or "0 0 0").split())
            t = joint_el.get("type")
            if t == "revolute":
                ax = [Q(v) for v in joint_el.find("axis").get("xyz").split()]
                assert ax == [0, 0, 1]
                jidx = int(joint_el.get("name").replace("panda_joint", "")) - 1
            elif t == "prismatic":
                ax = [Q(v) for v in joint_el.find("axis").get("xyz").split()]
                d = lock[joint_el.get("name")]
                # child = parent * T(xyz) * R(rpy) * T(d * axis): translation in parent = xyz + R(rpy) d axis
                Rr = rpy_exact_if_trivial(rpy)
                xyz = [xyz[r] + sum(Rr[r][k] * d * ax[k] for k in range(3)) for r in range(3)]
                jidx = -1
            elif t == "fixed":
                jidx = -1
            else:
                raise ValueError(t)
            xyz = tuple(xyz)
        m, c, I = inertial(name)
        out.append(OLink(name, parent_idx, jidx, xyz, rpy, m, c, I))
        for j in by_parent.get(name, []):
            ch = j.find("child").get("link")
            if ch in loaded:
                visit(ch, idx, j)
        return idx

    visit("panda_link0", -1, None)
    hand = names.index("panda_hand")
    m, c, I = Q("0.01"), (Q(0),) * 3, (Q("0.0001"),) * 3 + (Q(0),) * 3
    out.append(OLink("attached_object", hand, -1, (Q(0),) * 3, (Q(0),) * 3, m, c, I))
    gravity = (Q(0), Q(0), Q("-9.81"))
    return out, gravity


def rpy_exact_if_trivial(rpy):
    if any(v != 0 for v in rpy):
        raise ValueError("prismatic lock with rotated origin needs the rotation (not present in this URDF)")
    return [[Q(int(r == k)) for k in range(3)] for r in range(3)]


# ======================================================================== kinematics + energy
def _dot3(a, b):
    return _ad(_ad(_ml(a[0], b[0]), _ml(a[1], b[1])), _ml(a[2], b[2]))


def _mm(A, B):
    return [[_dot3(A[i], [B[0][k], B[1][k], B[2][k]]) for k in range(3)] for i in range(3)]


def _mv(A, v):
    return [_dot3(A[i], v) for i in range(3)]


class LagrangeOracle:
    def __init__(self, links, gravity, ctx: Ctx):
        self.links, self.ctx = links, ctx
        ctx.setup()
        _CTX[0] = ctx
        K = ctx.const
        self.g = [K(v) for v in gravity]
        self.Rfix, self.p, self.m, self.c, self.I = [], [], [], [], []
        for L in links:
            r, p_, y = (K(v) for v in L.rpy)
            cr, sr, cp, sp, cy, sy = ctx.cos(r), ctx.sin(r), ctx.cos(p_), ctx.sin(p_), ctx.cos(y), ctx.sin(y)
            z, o = K(0), K(1)
            Rx = [[o, z, z], [z, cr, -sr], [z, sr, cr]]
            Ry = [[cp, z, sp], [z, o, z], [-sp, z, cp]]
            Rz = [[cy, -sy, z], [sy, cy, z], [z, z, o]]
            self.Rfix.append(_mm(Rz, _mm(Ry, Rx)))
            self.p.append([K(v) for v in L.xyz])
            self.m.append(K(L.mass))
            self.c.append([K(v) for v in L.com])
            a = [K(v) for v in L.inertia]
            self.I.append([[a[0], a[3], a[4]], [a[3], a[1], a[5]], [a[4], a[5], a[2]]])
        self.dof = 1 + max(L.joint for L in links)

    def fk(self, q):
        """World rotation and CoM of every link for joint values q (scalars or Duals)."""
        _CTX[0] = self.ctx
        self.ctx.setup()
        n = len(self.links)
        K = self.ctx.const
        R = [None] * n
        o = [None] * n
        com = [None] * n
        for i, L in enumerate(self.links):
            if L.parent < 0:
                Rp = [[K(int(r == k)) for k in range(3)] for r in range(3)]
                op = [K(0)] * 3
            else:
                Rp, op = R[L.parent], o[L.parent]
            o[i] = [_ad(op[r], _dot3(Rp[r], self.p[i])) for r in range(3)]
            A = _mm(Rp, self.Rfix[i])
            if L.joint >= 0:
                cq, sq = self.ctx.cos(q[L.joint]), self.ctx.sin(q[L.joint])
                A = [[_ad(_ml(A[r][0], cq), _ml(A[r][1], sq)), _sb(_ml(A[r][1], cq), _ml(A[r][0], sq)), A[r][2]]
                     for r in range(3)]
            R[i] = A
            com[i] = [_ad(o[i][r], _dot3(A[r], self.c[i])) for r in range(3)]
        return R, com

    def T(self, q, v):
        """Kinetic energy at configuration q (scalars or outer Duals) and joint velocity v."""
        qi = [Dual(q[k], [v[k]]) for k in range(len(q))]
        R, com = self.fk(qi)
        tot = None
        for k in range(1, len(self.links)):
            vc = [com[k][r].d[0] if isinstance(com[k][r], Dual) else 0 for r in range(3)]
            Rv = [[dval(R[k][r][c]) for c in range(3)] for r in range(3)]
            Rd = [[R[k][r][c].d[0] if isinstance(R[k][r][c], Dual) else 0 for c in range(3)] for r in range(3)]
            # Omega^ = R^T Rdot ; Omega = (W[2][1], W[0][2], W[1][0])
            W = lambda a, b: _dot3([Rv[0][a], Rv[1][a], Rv[2][a]], [Rd[0][b], Rd[1][b], Rd[2][b]])  # noqa: E731
            Om = [W(2, 1), W(0, 2), W(1, 0)]
            IOm = _mv(self.I[k], Om)
            e = _ad(_ml(self.m[k], _dot3(vc, vc)), _dot3(Om, IOm))
            tot = e if tot is None else _ad(tot, e)
        return _ml(tot, self.ctx.const(Q(1, 2)))

    def V(self, q):
        _, com = self.fk(q)
        tot = None
        for k in range(1, len(self.links)):
            e = -_ml(self.m[k], _dot3(self.g, com[k]))
            tot = e if tot is None else _ad(tot, e)
        return tot

    # ------------------------------------------------------------------ dynamics
    def _unit(self, i):
        K = self.ctx.const
        return [K(int(k == i)) for k in range(self.dof)]

    def Mw(self, q, w):
        """M(q) w by polarization: (M w)_i = T(q, w+e_i) - T(q, w) - T(q, e_i)."""
        Tw = self.T(q, w)
        out = []
        for i in range(self.dof):
            e = self._unit(i)
            out.append(_sb(_sb(self.T(q, [_ad(w[k], e[k]) for k in range(self.dof)]), Tw), self.T(q, e)))
        return out

    def bias(self, q, qd):
        """c(q, qd) = Mdot qd - grad_q T(q, qd) by outer AD in directions e_0..e_6 and qd."""
        n = self.dof
        K = self.ctx.const
        dirs = [self._unit(i) for i in range(n)] + [list(qd)]
        qo = [Dual(q[k], [d[k] for d in dirs]) for k in range(n)]
        Tqd = self.T(qo, qd)
        gradT = [Tqd.d[i] for i in range(n)]
        Mqd_dot = []
        for i in range(n):
            e = self._unit(i)
            val = _sb(_sb(self.T(qo, [_ad(qd[k], e[k]) for k in range(n)]), Tqd), self.T(qo, e))
            Mqd_dot.append(val.d[n])
        del K
        return [Mqd_dot[i] - gradT[i] for i in range(n)]

    def gravity_torque(self, q):
        n = self.dof
        qo = [Dual(q[k], [self._unit(i)[k] for i in range(n)]) for k in range(n)]
        Vq = self.V(qo)
        return [Vq.d[i] for i in range(n)]

    def inverse_dynamics(self, q, qd, qdd, gravity=True):
        Mq = self.Mw(q, qdd)
        c = self.bias(q, qd)
        G = self.gravity_torque(q) if gravity else [self.ctx.const(0)] * self.dof
        return [Mq[i] + c[i] + G[i] for i in range(self.dof)]

    def abg(self, q, qp, qpp):
        """A = M q'' + c(q,q'), B = M q'/2, G = dV/dq (independent of the coefficient identities)."""
        half = self.ctx.const(Q(1, 2))
        Mpp = self.Mw(q, qpp)
        c = self.bias(q, qp)
        Mp = self.Mw(q, qp)
        G = self.gravity_torque(q)
        return ([Mpp[i] + c[i] for i in range(self.dof)], [Mp[i] * half for i in range(self.dof)], G)


# ======================================================================== helpers
def iv_bounds(v):
    """Exact Fraction endpoints of an mpmath iv interval (raw mpf tuples) or of an mpf point."""
    if hasattr(v, "_mpi_"):
        lo, hi = v._mpi_
        return _raw(lo), _raw(hi)
    x = _raw(v._mpf_)
    return x, x


def _raw(raw):
    sign, man, exp, bc = raw
    if man == 0:
        if raw == (0, 0, 0, 0):
            return Q(0)
        raise ValueError("special value")
    x = Q(int(man)) * Q(2) ** int(exp)
    return -x if sign else x


def verdict(lo: float, hi: float, oracle, tol=Q(0)):
    """CONTAINED / DISJOINT (proven when oracle is iv and tol=0) / AMBIGUOUS."""
    a, b = iv_bounds(oracle)
    a, b = a - tol, b + tol
    L, H = Q(float(lo)), Q(float(hi))
    if L <= a and b <= H:
        return "CONTAINED"
    if b < L or a > H:
        return "DISJOINT"
    return "AMBIGUOUS"


def path_raw(sp):
    """Exact power polynomials (local u) of q, q', q'', x, x' w.r.t. global s; own derivative formula."""
    h = Q(sp.right) - Q(sp.left)

    def der(p, k):
        return tuple(Q(c) * Q(math.factorial(n), math.factorial(n - k)) / h ** k
                     for n, c in enumerate(p) if n >= k) or (Q(0),)
    # shift index: der returns coefficients for u^(n-k)
    q = [tuple(Q(c) for c in p) for p in sp.q]
    return q, [der(p, 1) for p in q], [der(p, 2) for p in q], tuple(Q(c) for c in sp.x), der(tuple(sp.x), 1)


def peval(p, u):
    u = Q(u)
    return sum((Q(c) * u ** k for k, c in enumerate(p)), Q(0))


def state_at(sp, u):
    q, qp, qpp, x, xp = path_raw(sp)
    return ([peval(p, u) for p in q], [peval(p, u) for p in qp], [peval(p, u) for p in qpp],
            peval(x, u), peval(xp, u))


def path_torque(orc: LagrangeOracle, sp, u):
    """Oracle torque at local u with the PHYSICAL substitution qdot = q' sqrt(x),
    qddot = q'' x + q' x'/2 (sqrt in the oracle context; no affine identity used)."""
    q, qp, qpp, x, xp = state_at(sp, u)
    if x < 0:
        raise ValueError("x < 0")
    K = orc.ctx.const
    w = orc.ctx.sqrt(K(x))
    qd = [K(v) * w for v in qp]
    qdd = [K(a) * K(x) + K(b) * K(xp) * K(Q(1, 2)) for a, b in zip(qpp, qp)]
    return orc.inverse_dynamics([K(v) for v in q], qd, qdd)
