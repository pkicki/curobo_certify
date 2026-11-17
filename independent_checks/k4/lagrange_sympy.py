"""Second independent derivation: Euler-Lagrange equations by symbolic differentiation.

Requires sympy (optional; tests skip if missing). Positions of the CoMs are written
from planar geometry, kinetic energy T = sum 1/2 m |v_c|^2 + 1/2 J phidot^2,
potential V = -sum m g_vec . p_c, tau = d/dt dL/dqdot - dL/dq.
Evaluation is done with a reviewer-written expression-tree walker into mpmath.iv,
so sympy's own float printing is never used.
"""
from __future__ import annotations

from fractions import Fraction as Q

import sympy as sp
from mpmath import iv

from .oracle import riv, Chain, _ivctx


def derive(chain: Chain):
    n = chain.dof
    t = sp.Symbol("t")
    qf = [sp.Function(f"q{j}")(t) for j in range(n)]
    px, py, phi = sp.Integer(0), sp.Integer(0), sp.Integer(0)
    T, V = sp.Integer(0), sp.Integer(0)
    gx, gy = sp.Rational(chain.g[0]), sp.Rational(chain.g[1])
    for j, L in enumerate(chain.links):
        phi = phi + qf[j]
        cx = px + sp.Rational(L.c) * sp.cos(phi)
        cy = py + sp.Rational(L.c) * sp.sin(phi)
        vx, vy = sp.diff(cx, t), sp.diff(cy, t)
        T += sp.Rational(1, 2) * sp.Rational(L.m) * (vx ** 2 + vy ** 2) \
            + sp.Rational(1, 2) * sp.Rational(L.J) * sp.diff(phi, t) ** 2
        V += -sp.Rational(L.m) * (gx * cx + gy * cy)
        px = px + sp.Rational(L.l) * sp.cos(phi)
        py = py + sp.Rational(L.l) * sp.sin(phi)
    Lg = T - V
    qs = sp.symbols(f"qq0:{n}")
    vs = sp.symbols(f"vv0:{n}")
    as_ = sp.symbols(f"aa0:{n}")
    taus = []
    for j in range(n):
        e = sp.diff(sp.diff(Lg, sp.diff(qf[j], t)), t) - sp.diff(Lg, qf[j])
        rep = {}
        for k in range(n):
            rep[sp.diff(qf[k], t, 2)] = as_[k]
        e = e.subs(rep)
        rep = {sp.diff(qf[k], t): vs[k] for k in range(n)}
        e = e.subs(rep)
        e = e.subs({qf[k]: qs[k] for k in range(n)})
        taus.append(sp.expand(e))
    return qs, vs, as_, taus


def iv_eval(expr, env):
    """Reviewer-written walker: Rational/Integer exact -> riv, sin/cos -> iv."""
    _ivctx()
    if expr.is_Symbol:
        return env[expr]
    if expr.is_Integer or expr.is_Rational:
        return riv(Q(int(expr.p), int(expr.q)))
    if expr.is_Add:
        out = riv(0)
        for a in expr.args:
            out = out + iv_eval(a, env)
        return out
    if expr.is_Mul:
        out = riv(1)
        for a in expr.args:
            out = out * iv_eval(a, env)
        return out
    if expr.is_Pow:
        b, e = expr.args
        if not e.is_Integer or int(e) < 0:
            raise ValueError("unsupported power")
        base = iv_eval(b, env)
        out = riv(1)
        for _ in range(int(e)):
            out = out * base
        return out
    if expr.func is sp.sin:
        return iv.sin(iv_eval(expr.args[0], env))
    if expr.func is sp.cos:
        return iv.cos(iv_eval(expr.args[0], env))
    raise ValueError(f"unsupported node {expr.func}")


class LagrangeModel:
    def __init__(self, chain: Chain):
        self.chain = chain
        self.qs, self.vs, self.as_, self.taus = derive(chain)

    def F(self, q, v, a, w2=None):
        """tau(q, v, a); velocity-quadratic terms multiplied by w2 if given."""
        env = {}
        for k in range(len(self.qs)):
            env[self.qs[k]] = riv(q[k])
            env[self.vs[k]] = riv(v[k])
            env[self.as_[k]] = riv(a[k])
        if w2 is None:
            return [iv_eval(e, env) for e in self.taus]
        # velocities appear only quadratically: tau = tau(q,0,a) + w2*(tau(q,v,0)-tau(q,0,0))
        z = {**env, **{s: riv(0) for s in self.vs}}
        v0 = {**env, **{s: riv(0) for s in self.as_}}
        g = {**z, **{s: riv(0) for s in self.as_}}
        out = []
        for e in self.taus:
            out.append(iv_eval(e, z) + riv(w2) * (iv_eval(e, v0) - iv_eval(e, g)))
        return out

    def velocity_is_quadratic(self) -> bool:
        """Symbolic check that tau is a homogeneous quadratic in velocities + affine in a."""
        lam = sp.Symbol("lam")
        for e in self.taus:
            scaled = sp.expand(e.subs({v: lam * v for v in self.vs}, simultaneous=True))
            poly = sp.Poly(scaled, lam)
            if any(d not in (0, 2) for (d,) in poly.monoms()):
                return False
        return True
