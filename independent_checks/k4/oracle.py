"""Independent K4 P1 oracle (reviewer-written; imports NOTHING from src/curobo_certify).

Rigorous point evaluation of planar rigid-body inverse dynamics along a path q(s),
squared path speed x(s) = (ds/dt)^2, using

  * exact Fractions for every path quantity (own power-basis evaluation and own
    derivative formula d^k/ds^k sum c_n u^n = sum c_n n!/(n-k)! u^(n-k) / h^k);
  * mpmath.iv (outward-rounded interval arithmetic, rigorous sin/cos) for all
    trigonometric / dynamics arithmetic, with exact Fraction <-> mpf conversions;
  * a planar projected Newton-Euler (Kane / d'Alembert) formulation:
        tau_j = sum_{i>=j} [ perp(pc_i - p_j) . m_i (a_ci - g_vec) + J_i alpha_i ]
    with recursive planar point accelerations. This is NOT the closed-form
    M/C/g formula of the reviewed code; the closed form is only *compared* to it.

Velocity substitution along the path:  qdot_j = q'_j w,  w^2 = x,
  qddot_j = q''_j x + q'_j x'/2. Products qdot_k qdot_l = q'_k q'_l x, so no sqrt is
needed and x = 0 is handled exactly.

Conventions mapped to the reviewed fixtures (and independently re-derived here):
  * two-link: absolute link angles phi_1 = q1, phi_2 = q1 + q2 measured from +x,
    gravity vector (0, -g); J_i are CoM inertias.
  * pendulum: one link, angle measured from the downward vertical, which in this
    planar chain is equivalent to gravity vector (+g, 0); the reviewed `inertia`
    parameter is the PIVOT inertia, so J_com = inertia - m c^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Sequence

import mpmath  # noqa: F401  (iv context)
from mpmath import iv

PREC = 256
_PREC_LADDER = (256, 640, 1600)


def _ivctx():
    iv.prec = PREC
    return iv


def set_prec(bits: int):
    global PREC
    PREC = int(bits)
    iv.prec = PREC


# ---------------------------------------------------------------- conversions
def _raw_to_q(raw) -> Q:
    sign, man, exp, bc = raw
    if man == 0:
        if raw == (0, 0, 0, 0):
            return Q(0)
        raise ValueError("nonfinite/special interval endpoint")
    v = Q(int(man)) * (Q(2) ** int(exp))
    return -v if sign else v


def iv_bounds(v) -> tuple[Q, Q]:
    """Exact Fraction endpoints of an mpmath.iv interval (read from raw mpf tuples)."""
    lo, hi = v._mpi_
    return _raw_to_q(lo), _raw_to_q(hi)


def riv(fr) -> "iv.mpf":
    """Rigorous interval enclosing the exact rational fr."""
    _ivctx()
    fr = Q(fr)
    v = iv.mpf(fr.numerator) / iv.mpf(fr.denominator)
    lo, hi = iv_bounds(v)
    if not (lo <= fr <= hi):  # self-check of the conversion itself
        raise AssertionError("rational->interval conversion lost the value")
    return v


# ---------------------------------------------------------------- polynomials
def poly_eval(c: Sequence[Q], u: Q) -> Q:
    """Plain power sum (deliberately not Horner)."""
    u = Q(u)
    return sum((Q(ck) * u ** k for k, ck in enumerate(c)), Q(0))


def poly_deriv_eval(c: Sequence[Q], u: Q, k: int, h: Q) -> Q:
    """d^k/ds^k of sum c_n u^n with u=(s-left)/h, evaluated at u, exactly."""
    u = Q(u)
    h = Q(h)
    tot = Q(0)
    for n, cn in enumerate(c):
        if n < k:
            continue
        tot += Q(cn) * Q(math.factorial(n), math.factorial(n - k)) * u ** (n - k)
    return tot / h ** k


def shift_poly(c: Sequence[Q], left: Q, h: Q) -> tuple[Q, ...]:
    """Coefficients (in u) of P(left + h u) for a global power-basis P(s)."""
    out = [Q(0)] * len(c)
    for n, cn in enumerate(c):
        for k in range(n + 1):
            out[k] += Q(cn) * math.comb(n, k) * Q(left) ** (n - k) * Q(h) ** k
    return tuple(out)


def bernstein_identity_holds(power: Sequence[Q], bern: Sequence[Q]) -> bool:
    """Exact polynomial identity sum b_k C(n,k) u^k(1-u)^(n-k) == sum p_i u^i."""
    n = len(bern) - 1
    if len(power) > n + 1:
        return False
    coeff = [Q(0)] * (n + 1)
    for k, bk in enumerate(bern):
        # C(n,k) u^k (1-u)^(n-k) = C(n,k) sum_j C(n-k,j)(-1)^j u^(k+j)
        for j in range(n - k + 1):
            coeff[k + j] += Q(bk) * math.comb(n, k) * math.comb(n - k, j) * (-1) ** j
    target = [Q(p) for p in power] + [Q(0)] * (n + 1 - len(power))
    return coeff == target


def path_interval(path: str) -> tuple[Q, Q]:
    """Own decoding of an L/R dyadic path to [a,b] subset [0,1]."""
    a = Q(0)
    for i, ch in enumerate(path):
        if ch == "R":
            a += Q(1, 2 ** (i + 1))
        elif ch != "L":
            raise ValueError("bad path char")
    return a, a + Q(1, 2 ** len(path))


# ---------------------------------------------------------------- models
@dataclass(frozen=True)
class Link:
    m: Q
    c: Q   # CoM distance from proximal joint along link
    l: Q   # distance to next joint
    J: Q   # inertia about CoM


@dataclass(frozen=True)
class Chain:
    links: tuple[Link, ...]
    g: tuple[Q, Q]   # gravity acceleration vector

    @property
    def dof(self):
        return len(self.links)


def pendulum_chain(inertia, mass, com, gravity) -> Chain:
    inertia, mass, com, gravity = map(Q, (inertia, mass, com, gravity))
    return Chain((Link(mass, com, Q(0), inertia - mass * com * com),), (gravity, Q(0)))


def two_link_chain(m1, m2, l1, c1, c2, J1, J2, gravity) -> Chain:
    m1, m2, l1, c1, c2, J1, J2, gravity = map(Q, (m1, m2, l1, c1, c2, J1, J2, gravity))
    return Chain((Link(m1, c1, l1, J1), Link(m2, c2, Q(0), J2)), (Q(0), -gravity))


def inverse_dynamics(chain: Chain, q, v, a, w2=None):
    """Kane/Newton-Euler planar inverse dynamics with rigorous intervals.

    q, v, a: sequences of iv (or Fractions). Velocity products are scaled by w2
    (w2 = 1 gives ordinary F(q, v, a); w2 = x gives the path substitution).
    """
    _ivctx()
    cv = lambda z: z if isinstance(z, iv.mpf) else riv(z)
    q = [cv(z) for z in q]
    v = [cv(z) for z in v]
    a = [cv(z) for z in a]
    w2 = riv(1) if w2 is None else cv(w2)
    zero = riv(0)
    n = chain.dof
    p = [(zero, zero)]          # joint positions
    pc, ac, alpha = [], [], []
    ap = (zero, zero)
    phi = zero
    om = zero
    al = zero
    for i, L in enumerate(chain.links):
        phi = phi + q[i]
        om = om + v[i]
        al = al + a[i]
        e = (iv.cos(phi), iv.sin(phi))
        nn = (-e[1], e[0])
        acc_dir = (al * nn[0] - om * om * w2 * e[0], al * nn[1] - om * om * w2 * e[1])
        cL, lL = riv(L.c), riv(L.l)
        pc.append((p[i][0] + cL * e[0], p[i][1] + cL * e[1]))
        ac.append((ap[0] + cL * acc_dir[0], ap[1] + cL * acc_dir[1]))
        alpha.append(al)
        p.append((p[i][0] + lL * e[0], p[i][1] + lL * e[1]))
        ap = (ap[0] + lL * acc_dir[0], ap[1] + lL * acc_dir[1])
    gx, gy = riv(chain.g[0]), riv(chain.g[1])
    tau = []
    for j in range(n):
        t = zero
        for i in range(j, n):
            L = chain.links[i]
            rx, ry = pc[i][0] - p[j][0], pc[i][1] - p[j][1]
            perp = (-ry, rx)
            m = riv(L.m)
            fx, fy = m * (ac[i][0] - gx), m * (ac[i][1] - gy)
            t = t + perp[0] * fx + perp[1] * fy + riv(L.J) * alpha[i]
        tau.append(t)
    return tau


# ---------------------------------------------------------------- path
@dataclass(frozen=True)
class OSpan:
    left: Q
    right: Q
    q: tuple[tuple[Q, ...], ...]
    x: tuple[Q, ...]

    @property
    def h(self):
        return Q(self.right) - Q(self.left)


def path_state(sp: OSpan, u: Q):
    h = sp.h
    q = [poly_eval(p, u) for p in sp.q]
    qp = [poly_deriv_eval(p, u, 1, h) for p in sp.q]
    qpp = [poly_deriv_eval(p, u, 2, h) for p in sp.q]
    x = poly_eval(sp.x, u)
    xp = poly_deriv_eval(sp.x, u, 1, h)
    return q, qp, qpp, x, xp


def path_torque(chain: Chain, sp: OSpan, u: Q):
    """Rigorous enclosure of the inverse-dynamics torque at local u (x(u) >= 0)."""
    q, qp, qpp, x, xp = path_state(sp, Q(u))
    if x < 0:
        raise ValueError("x(u) < 0: torque undefined")
    acc = [qpp[j] * x + qp[j] * xp / 2 for j in range(len(q))]  # exact rationals
    return inverse_dynamics(chain, q, qp, acc, w2=x)


def exact_torque_if_trig_free(chain: Chain, sp: OSpan, u: Q):
    """Exact rational torque when the dynamics provably needs no trig value.

    Single link with zero gravity: from the Kane sum, perp(c e).m(c(alpha n - w^2 e))
    = m c^2 alpha (n.n = 1, n.e = 0), so tau = (J + m c^2) * qddot exactly.
    Two-link with zero gravity and q2 == 0 identically on the evaluation point: angles
    enter only through q2 (cos 0 = 1, sin 0 = 0), handled by an exact recomputation.
    Returns None when not applicable.
    """
    q, qp, qpp, x, xp = path_state(sp, Q(u))
    if x < 0 or chain.g != (Q(0), Q(0)):
        return None
    acc = [qpp[j] * x + qp[j] * xp / 2 for j in range(len(q))]
    if chain.dof == 1:
        L = chain.links[0]
        return [(L.J + L.m * L.c * L.c) * acc[0]]
    if chain.dof == 2 and q[1] == 0:
        L1, L2 = chain.links
        # rotate frame by -q1 (rotation invariance of the planar dot products), q2 = 0:
        # every unit vector is (1,0)/(0,1) exactly; re-run the Kane recursion in rationals.
        w1, w2_ = qp[0], qp[0] + qp[1]
        a1, a2 = acc[0], acc[0] + acc[1]
        e, n = (Q(1), Q(0)), (Q(0), Q(1))
        d1 = (a1 * n[0] - w1 * w1 * x * e[0], a1 * n[1] - w1 * w1 * x * e[1])
        d2 = (a2 * n[0] - w2_ * w2_ * x * e[0], a2 * n[1] - w2_ * w2_ * x * e[1])
        p1 = (L1.l, Q(0))
        pc = [(L1.c, Q(0)), (L1.l + L2.c, Q(0))]
        ac = [(L1.c * d1[0], L1.c * d1[1]),
              (L1.l * d1[0] + L2.c * d2[0], L1.l * d1[1] + L2.c * d2[1])]
        joints = [(Q(0), Q(0)), p1]
        alph = [a1, a2]
        out = []
        for j in range(2):
            t = Q(0)
            for i in range(j, 2):
                Li = chain.links[i]
                rx, ry = pc[i][0] - joints[j][0], pc[i][1] - joints[j][1]
                t += -ry * Li.m * ac[i][0] + rx * Li.m * ac[i][1] + Li.J * alph[i]
            out.append(t)
        return out
    return None


def abg(chain: Chain, q, qp, qpp):
    """Coefficient identities G=F(q,0,0), B=(F(q,0,q')-G)/2, A=F(q,q',q'')-G."""
    n = len(q)
    z = [Q(0)] * n
    G = inverse_dynamics(chain, q, z, z)
    Fb = inverse_dynamics(chain, q, z, qp)
    Fa = inverse_dynamics(chain, q, qp, qpp)
    half = riv(Q(1, 2))
    return ([Fa[j] - G[j] for j in range(n)], [(Fb[j] - G[j]) * half for j in range(n)], G)


# ---------------------------------------------------------------- verdicts
def containment(box_lo: Q, box_hi: Q, oracle) -> str:
    """CONTAINED: oracle interval inside box (=> true value inside box, proven).
    DISJOINT: oracle interval misses box (=> true value outside box, proven: counterexample).
    AMBIGUOUS: overlap only (no conclusion)."""
    lo, hi = iv_bounds(oracle)
    if box_lo <= lo and hi <= box_hi:
        return "CONTAINED"
    if hi < box_lo or lo > box_hi:
        return "DISJOINT"
    return "AMBIGUOUS"


def check_box(box_lo: Q, box_hi: Q, compute) -> str:
    """Adaptive-precision containment of a reviewed box against an oracle quantity.

    compute() must return the oracle iv enclosure (re-evaluated at the current PREC).
    Returns CONTAINED / DISJOINT / POINT_CONSISTENT (reviewed box is a single exact
    rational lying inside the oracle enclosure at the highest precision) /
    EDGE_CONSISTENT (a box endpoint lies inside the 1600-bit oracle enclosure, i.e. the
    true value equals the endpoint to within 2^-1590 relative; NOT a proof, reported) /
    AMBIGUOUS.
    """
    global PREC
    old = PREC
    try:
        verdict = "AMBIGUOUS"
        for bits in _PREC_LADDER:
            set_prec(bits)
            o = compute()
            verdict = containment(box_lo, box_hi, o)
            if verdict in ("CONTAINED", "DISJOINT"):
                return verdict
        lo, hi = iv_bounds(o)
        if box_lo == box_hi:
            if lo <= box_lo <= hi:
                return "POINT_CONSISTENT"
        elif lo <= box_lo <= hi or lo <= box_hi <= hi:
            return "EDGE_CONSISTENT"
        return verdict
    finally:
        set_prec(old)


def verify_violation(chain: Chain, sp: OSpan, u: Q, joint: int, limit: tuple[Q, Q]) -> bool:
    """Independent witness check: true torque at u is PROVABLY outside [lo,hi]."""
    if poly_eval(sp.x, u) < 0:
        return False
    t = path_torque(chain, sp, u)[joint]
    lo, hi = iv_bounds(t)
    return lo > Q(limit[1]) or hi < Q(limit[0])


def true_trig_range_inside(lo: Q, hi: Q, enc_lo: Q, enc_hi: Q, cosine: bool) -> str:
    """Is the exact range of sin/cos over [lo,hi] inside [enc_lo, enc_hi]?

    The exact range is the hull of the endpoint values and of +-1 at every
    stationary point in [lo,hi]. Stationary points are k*pi + pi/2 (sin) or
    k*pi (cos); any whose rigorous pi-enclosure merely touches [lo,hi] is included
    (stricter test). Returns CONTAINED / DISJOINT(counterexample) / AMBIGUOUS.
    """
    _ivctx()
    f = iv.cos if cosine else iv.sin
    cands = [f(riv(lo)), f(riv(hi))]
    pi = iv.pi
    lof, hif = float(lo), float(hi)
    kmin = math.floor(lof / math.pi) - 2
    kmax = math.ceil(hif / math.pi) + 2
    for k in range(kmin, kmax + 1):
        sp = pi * k + (0 if cosine else pi / 2)
        sa, sb = iv_bounds(sp)
        if sb < lo or sa > hi:
            continue
        val = (-1) ** k if cosine else (-1) ** k  # cos(k pi) = (-1)^k ; sin(k pi + pi/2) = (-1)^k
        cands.append(riv(val))
    worst = "CONTAINED"
    for c in cands:
        v = containment(enc_lo, enc_hi, c)
        if v == "DISJOINT":
            return "DISJOINT"
        if v == "AMBIGUOUS":
            worst = "AMBIGUOUS"
    return worst


# ---------------------------------------------------------------- random families
def random_c2_spline(rng, dof: int, knots: Sequence[Q], qscale: Q = Q(3), xbase=None,
                     allow_negative_x=False):
    """Piecewise cubic q (C2 at knots, genuine 3rd-derivative jumps) and cubic x (C1).

    q_j(s) on span k: P_j(s) + sum_{knots t_i < s} d_ji (s - t_i)^3 ; same trick with
    (s - t_i)^2 for x gives C1 with jumping x''.
    Returns list of OSpan in local coordinates.
    """
    def rq(scale, den=8):
        return Q(rng.randint(-int(scale * den), int(scale * den)), den)

    P = [[rq(qscale) for _ in range(4)] for _ in range(dof)]
    jumps = [[rq(qscale) for _ in knots[1:-1]] for _ in range(dof)]
    X = [rq(2) for _ in range(4)]
    xj = [rq(1) for _ in knots[1:-1]]
    spans = []
    for k in range(len(knots) - 1):
        left, right = Q(knots[k]), Q(knots[k + 1])
        h = right - left
        qs = []
        for j in range(dof):
            gl = list(P[j])
            for i, t in enumerate(knots[1:k + 1]):
                d = jumps[j][i]
                t = Q(t)
                # d (s - t)^3 = d (s^3 - 3 t s^2 + 3 t^2 s - t^3)
                for n, c in enumerate((-t ** 3, 3 * t ** 2, -3 * t, Q(1))):
                    gl[n] += d * c
            qs.append(shift_poly(gl, left, h))
        gx = list(X)
        for i, t in enumerate(knots[1:k + 1]):
            t = Q(t)
            for n, c in enumerate((t ** 2, -2 * t, Q(1))):
                gx[n] += xj[i] * c
        spans.append([left, right, tuple(qs), gx])
    # make x >= 0 (or deliberately not) by adding a constant computed exactly from
    # a crude rigorous lower bound: |x(s)| <= sum |coeffs| for s in [0,1] (knots in [0,1]).
    if xbase is None:
        bound = max(sum(abs(c) for c in s[3]) for s in spans)
        shift = bound + Q(rng.randint(0, 8), 8) if not allow_negative_x else Q(rng.randint(-8, 8), 8)
    else:
        shift = Q(xbase)
    out = []
    for left, right, qs, gx in spans:
        gx = list(gx)
        gx[0] += shift
        out.append(OSpan(left, right, qs, shift_poly(gx, left, right - left)))
    return out
