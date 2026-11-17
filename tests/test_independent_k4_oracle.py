"""P1 independent review of the K4 analytic torque reference (reviewer-written).

Run:  PYTHONPATH=src:. python -m unittest tests.test_independent_k4_oracle -v
(also picked up by `python -m unittest discover -s tests -p 'test_*.py'`; this file
inserts src/ and the repository root into sys.path itself).

Truth comes from independent_checks/k4/oracle.py: a planar Kane/Newton-Euler
inverse-dynamics recursion in mpmath.iv (rigorous sin/cos, outward rounding) with
exact-Fraction path evaluation, and (if sympy is installed) a second, symbolic
Euler-Lagrange derivation. Nothing here imports the reviewed dynamics formulas into
the oracle; the reviewed code is only called and its outputs are compared.

A DISJOINT verdict (oracle enclosure strictly outside a reviewed box) is a proven
counterexample. Dense sampling is falsification evidence, not a proof of soundness.
"""
from __future__ import annotations

import ast
import random
import sys
import unittest
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import mpmath  # noqa: F401
    HAVE_MPMATH = True
except ImportError:  # pragma: no cover
    HAVE_MPMATH = False
try:
    import sympy  # noqa: F401
    HAVE_SYMPY = True
except ImportError:  # pragma: no cover
    HAVE_SYMPY = False

from curobo_certify import k4_reference as R  # noqa: E402
from curobo_certify.k4_reference import (I, Pendulum, PlanarTwoLink, PathSpan, K4Result,  # noqa: E402
                                         certify_torque, replay_torque_leaves, isin, icos)
from curobo_certify import k4_affine as AF  # noqa: E402

if HAVE_MPMATH:
    from independent_checks.k4 import oracle as O  # noqa: E402
    from independent_checks.k4 import campaign as C  # noqa: E402

P10 = F(1, 10 ** 30)


def ospan(left, right, q, x):
    return O.OSpan(F(left), F(right), tuple(tuple(F(c) for c in p) for p in q), tuple(F(c) for c in x))


def cspans(osps):
    return [PathSpan(s.left, s.right, s.q, s.x) for s in osps]


def peak_x(P, K, c, X0=10):
    """x(u) with x'(u)/2 = P - K (u-c)^2 (tau for pendulum I=1, g=0, q=u)."""
    P, K, c, X0 = map(F, (P, K, c, X0))
    return (X0, 2 * P - 2 * K * c * c, 2 * K * c, -2 * K / 3)


UNIT_PEND = dict(inertia=1, mass=1, com=1, gravity=0)   # pivot inertia = m c^2 (point mass)


@unittest.skipUnless(HAVE_MPMATH, "mpmath required for the independent oracle")
class OracleSelfChecks(unittest.TestCase):
    def test_rational_iv_conversion_is_enclosing(self):
        rng = random.Random(1)
        for _ in range(200):
            fr = F(rng.randint(-10 ** 40, 10 ** 40), rng.randint(1, 10 ** 35))
            lo, hi = O.iv_bounds(O.riv(fr))
            self.assertLessEqual(lo, fr)
            self.assertLessEqual(fr, hi)

    def test_exact_trig_free_branch_matches_interval_oracle(self):
        rng = random.Random(2)
        for _ in range(40):
            ch = O.two_link_chain(*(F(rng.randint(1, 30), 10) for _ in range(5)),
                                  F(rng.randint(0, 5), 10), F(rng.randint(0, 5), 10), 0)
            sp = ospan(0, F(rng.randint(1, 4), 2),
                       [[F(rng.randint(-9, 9), 4) for _ in range(4)], [0]],
                       [5, F(rng.randint(-9, 9), 8), F(rng.randint(-9, 9), 8)])
            u = F(rng.randint(0, 16), 16)
            ex = O.exact_torque_if_trig_free(ch, sp, u)
            for j, t in enumerate(O.path_torque(ch, sp, u)):
                lo, hi = O.iv_bounds(t)
                self.assertTrue(lo <= ex[j] <= hi)

    @unittest.skipUnless(HAVE_SYMPY, "sympy optional second derivation")
    def test_kane_oracle_matches_symbolic_lagrangian(self):
        from independent_checks.k4.lagrange_sympy import LagrangeModel
        rng = random.Random(3)
        chains = [O.two_link_chain(F(3, 2), 1, F(7, 10), F(1, 3), F(2, 5), F(1, 10), F(1, 20), F(981, 100)),
                  O.two_link_chain(2, F(1, 2), F(3, 2), F(3, 4), F(1, 5), 0, F(1, 7), F(7, 2)),
                  O.pendulum_chain(2, 1, F(1, 2), F(981, 100))]
        for ch in chains:
            L = LagrangeModel(ch)
            self.assertTrue(L.velocity_is_quadratic())
            for _ in range(15):
                q, v, a = ([F(rng.randint(-40, 40), 7) for _ in range(ch.dof)] for _ in range(3))
                for t1, t2 in zip(O.inverse_dynamics(ch, q, v, a), L.F(q, v, a)):
                    a1, b1 = O.iv_bounds(t1)
                    a2, b2 = O.iv_bounds(t2)
                    self.assertTrue(a1 <= b2 and a2 <= b1, "two independent oracles disagree")
                    self.assertLess(b1 - a1, F(1, 10 ** 50))


@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class TrigAudit(unittest.TestCase):
    def _verdict(self, lo, hi, enc, cosine):
        for bits in O._PREC_LADDER:
            O.set_prec(bits)
            v = O.true_trig_range_inside(lo, hi, enc.lo, enc.hi, cosine)
            if v != "AMBIGUOUS":
                break
        O.set_prec(256)
        return v

    def test_sin_cos_enclose_exact_range_random_including_large(self):
        rng = random.Random(11)
        bad, amb = [], 0
        for _ in range(500):
            c = F(rng.randint(-2400, 2400), 100) + F(rng.randint(0, 10 ** 6), 10 ** 9)
            r = rng.choice([F(0), F(1, 10 ** 6), F(1, 100), F(1, 3), F(2), F(7)])
            for f, cosine in ((isin, False), (icos, True)):
                enc = f(I(c - r, c + r))
                v = self._verdict(c - r, c + r, enc, cosine)
                if v == "DISJOINT":
                    bad.append((str(c), str(r), cosine))
                amb += v == "AMBIGUOUS"
        self.assertEqual(bad, [])
        self.assertEqual(amb, 0)

    def test_taylor_lagrange_remainder_at_threshold(self):
        # |c|=16 is the worst admitted midpoint; Lagrange term 16^57/57! ~ 1.06e-8 (cos), 16^58/58! (sin)
        for c in (F(16), F(-16), F(16) - F(1, 10 ** 20), F(31, 2)):
            for cosine in (False, True):
                enc = R._sin_cos_midpoint(c, cosine=cosine)
                self.assertNotEqual(self._verdict(c, c, enc, cosine), "DISJOINT")
                self.assertLess(enc.hi - enc.lo, F(3, 10 ** 8))

    def test_fallback_boundary_and_huge_angles(self):
        self.assertNotEqual(isin(I.point(16)), I(-1, 1))           # mid == 16 uses Taylor
        self.assertEqual(isin(I.point(F(16) + P10)), I(-1, 1))     # mid > 16 falls back
        self.assertEqual(icos(I(-10 ** 9, -10 ** 9 + 1)), I(-1, 1))
        # intervals whose midpoint is <=16 but which reach far beyond: Lipschitz widening clamps
        enc = isin(I(F(-100), F(130)))   # mid 15
        self.assertEqual(enc, I(-1, 1))

    def test_lipschitz_widening_across_stationary_points(self):
        for lo, hi in ((F(1), F(2)), (F(3), F(F(32, 10))), (F(-2), F(2)), (F(15), F(17)), (F(6), F(13, 2))):
            for f, cosine in ((isin, False), (icos, True)):
                self.assertNotEqual(self._verdict(lo, hi, f(I(lo, hi)), cosine), "DISJOINT")


class IntervalArithmeticAudit(unittest.TestCase):
    CLASSES = [(-5, -2), (-3, 0), (-4, 6), (0, 0), (0, 3), (2, 7), (-F(7, 3), F(1, 9))]

    def test_multiplication_all_sign_cases_exact_range(self):
        rng = random.Random(5)
        for a in self.CLASSES:
            for b in self.CLASSES:
                A, B = I(*a), I(*b)
                P = A * B
                self.assertIs(type(P.lo), F)
                # attained: endpoints are products of endpoints (tight) ...
                prods = {A.lo * B.lo, A.lo * B.hi, A.hi * B.lo, A.hi * B.hi}
                self.assertIn(P.lo, prods)
                self.assertIn(P.hi, prods)
                # ... and every sampled product lies inside
                for _ in range(200):
                    x = A.lo + (A.hi - A.lo) * F(rng.randint(0, 64), 64)
                    y = B.lo + (B.hi - B.lo) * F(rng.randint(0, 64), 64)
                    self.assertTrue(P.lo <= x * y <= P.hi)

    def test_division_and_zero_cases(self):
        rng = random.Random(6)
        for a in self.CLASSES:
            for b in ((1, 3), (-4, -F(1, 2)), (F(1, 7), F(1, 7))):
                A, B = I(*a), I(*b)
                D = A / B
                for _ in range(200):
                    x = A.lo + (A.hi - A.lo) * F(rng.randint(0, 64), 64)
                    y = B.lo + (B.hi - B.lo) * F(rng.randint(0, 64), 64)
                    self.assertTrue(D.lo <= x / y <= D.hi)
        for z in ((-1, 1), (0, 1), (-1, 0), (0, 0)):
            with self.assertRaises(ZeroDivisionError):
                _ = I(1, 2) / I(*z)

    def test_no_float_literals_or_float_calls_in_reviewed_modules(self):
        for rel in ("src/curobo_certify/k4_reference.py", "src/curobo_certify/k4_affine.py"):
            tree = ast.parse((ROOT / rel).read_text())
            floats = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
            calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id in ("float", "round")]
            mathfn = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "math"
                      and any(a.name not in ("comb", "factorial", "isfinite") for a in n.names)]
            self.assertEqual((floats, calls, mathfn), ([], [], []), rel)

    def test_float_inputs_are_converted_exactly_not_decimally(self):
        # Documented behaviour (scope note): floats are accepted and converted to their
        # exact binary value; 0.1 is NOT 1/10. No untracked float arithmetic follows.
        sp = PathSpan.from_values(0, 0.1, [[0.1]], [1])
        self.assertEqual(sp.right, F(0.1))
        self.assertNotEqual(sp.right, F(1, 10))
        self.assertIs(type(sp.q[0][0]), F)
        with self.assertRaises(ValueError):
            PathSpan.from_values(0, float("inf"), [[0]], [1])
        r = certify_torque([PathSpan.from_values(0, 1, [[0.5, 0.25]], [1.5])], Pendulum(1, 1, 1, 1), [(-2, 2)])
        for lf in r.leaves:
            for t in lf.torque:
                self.assertIs(type(t.lo), F)
                self.assertIs(type(t.hi), F)


@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class PolynomialAudit(unittest.TestCase):
    def test_power_to_bernstein_identity(self):
        rng = random.Random(7)
        for n in range(0, 7):
            for _ in range(30):
                p = tuple(F(rng.randint(-99, 99), rng.randint(1, 9)) for _ in range(n + 1))
                self.assertTrue(O.bernstein_identity_holds(p, R.bernstein(p)))

    def test_de_casteljau_split_identity(self):
        rng = random.Random(8)
        for n in range(0, 7):
            for _ in range(30):
                p = tuple(F(rng.randint(-99, 99), rng.randint(1, 9)) for _ in range(n + 1))
                left, right = R.split(R.bernstein(p))
                self.assertTrue(O.bernstein_identity_holds(O.shift_poly(p, 0, F(1, 2)), left))
                self.assertTrue(O.bernstein_identity_holds(O.shift_poly(p, F(1, 2), F(1, 2)), right))

    def test_dpoly_scaling_by_span_length(self):
        rng = random.Random(9)
        for _ in range(200):
            p = tuple(F(rng.randint(-99, 99), rng.randint(1, 9)) for _ in range(rng.randint(1, 4)))
            h = F(rng.randint(1, 50), rng.randint(1, 50))
            u = F(rng.randint(0, 100), 100)
            for k in range(4):
                self.assertEqual(R.eval_poly(R.dpoly(p, k, h), u), O.poly_deriv_eval(p, u, k, h))

    def test_compose_subinterval_matches_own_shift(self):
        rng = random.Random(10)
        for _ in range(100):
            p = tuple(F(rng.randint(-99, 99), rng.randint(1, 9)) for _ in range(4))
            a = F(rng.randint(0, 8), 16)
            b = a + F(rng.randint(1, 8), 16)
            self.assertEqual(R.compose_subinterval(p, a, b), O.shift_poly(p, a, b - a))


@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class DynamicsAudit(unittest.TestCase):
    def _cmp(self, model, chain, q, d, dd):
        A, B, G = model.coefficients(tuple(I.point(z) for z in q), tuple(I.point(z) for z in d),
                                     tuple(I.point(z) for z in dd))
        out = []
        for k, coeffs in enumerate((A, B, G)):
            for j, box in enumerate(coeffs):
                out.append(O.check_box(box.lo, box.hi, lambda k=k, j=j: O.abg(chain, q, d, dd)[k][j]))
        return out

    def test_two_link_ABG_vs_kane_oracle_random(self):
        rng = random.Random(12)
        verdicts = []
        for _ in range(150):
            par = C.rand_two_link(rng)
            par["gravity"] = rng.choice([F(981, 100), F(rng.randint(1, 200), 10)])
            model, chain = PlanarTwoLink(**par), O.two_link_chain(**par)
            q = [F(rng.randint(-1500, 1500), 97) for _ in range(2)]
            if q[1] == 0:
                q[1] = F(1, 3)
            d = [F(rng.randint(-60, 60), 7) for _ in range(2)]
            dd = [F(rng.randint(-60, 60), 7) for _ in range(2)]
            verdicts += self._cmp(model, chain, q, d, dd)
        self.assertNotIn("DISJOINT", verdicts)
        self.assertNotIn("AMBIGUOUS", verdicts)
        self.assertNotIn("EDGE_CONSISTENT", verdicts)

    def test_pendulum_ABG_vs_kane_oracle_random(self):
        rng = random.Random(13)
        verdicts = []
        for _ in range(100):
            par = C.rand_pendulum(rng)
            verdicts += self._cmp(Pendulum(**par), O.pendulum_chain(**par),
                                  [F(rng.randint(-1500, 1500), 97)], [F(rng.randint(-60, 60), 7)],
                                  [F(rng.randint(-60, 60), 7)])
        self.assertTrue(set(verdicts) <= {"CONTAINED", "POINT_CONSISTENT"}, set(verdicts))

    def test_coriolis_and_gravity_hand_values(self):
        # q2 = 1, q1' = 1, q2' = 0, q'' = 0, g = 0: tau2 = +m2 l1 c2 sin(q2) q1'^2 (centrifugal), tau1 = 0
        m = PlanarTwoLink(1, 2, F(3, 2), F(1, 2), F(1, 3), 0, 0, 0)
        A, B, G = m.coefficients((I.point(0), I.point(1)), (I.point(1), I.point(0)), (I.point(0), I.point(0)))
        self.assertEqual(A[0], I.point(0))
        self.assertTrue(A[1].lo > F(8, 10) and A[1].hi < F(85, 100))   # 2*1.5*(1/3)*sin(1)=0.8415
        # horizontal outstretched arm under gravity: tau = g (m1 c1 + m2 (l1 + c2)), g m2 c2
        m = PlanarTwoLink(1, 2, F(3, 2), F(1, 2), F(1, 3), 0, 0, 10)
        _, _, G = m.coefficients((I.point(0), I.point(0)), (I.point(0),) * 2, (I.point(0),) * 2)
        self.assertEqual(G, (I.point(10 * (F(1, 2) + 2 * (F(3, 2) + F(1, 3)))), I.point(10 * 2 * F(1, 3))))

    def test_pendulum_inertia_is_pivot_and_unphysical_values_accepted(self):
        # SCOPE FINDING: `inertia` is the pivot inertia (oracle needs J_com = inertia - m c^2).
        # Pendulum(inertia=0, mass=1, com=1) is accepted although no rigid body has
        # pivot inertia < m c^2. Certificates are then for an unphysical model.
        p = Pendulum(0, 1, 1, 1)
        self.assertEqual(p.coefficients((I.point(0),), (I.point(1),), (I.point(1),))[0][0], I.point(0))


def audit(test, chain, osps, limits, result):
    st = C.audit_result(random.Random(99), chain, osps, limits, result, extra_per_leaf=4, dense=96)
    test.assertEqual(st["disjoint"], [], "COUNTEREXAMPLE: reviewed claim contradicted by oracle")
    test.assertEqual(st["ambiguous"], 0)
    return st


@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class PathCertificationAdversarial(unittest.TestCase):
    def test_randomized_rational_families_no_counterexample(self):
        s = C.run_campaign(20261117, 80, max_depth=6, max_nodes=400, dense=48, extra=2, affine_rate=0.5)
        self.assertEqual(s["counterexamples"], [])
        self.assertEqual(s["ambiguous"], 0)
        self.assertEqual(s["replay_rejected_certified"], 0)
        self.assertEqual(s["affine_replay_rejected"], 0)
        self.assertGreater(s["status"].get("CERTIFIED", 0), 5)
        self.assertGreater(s["status"].get("VIOLATED_TORQUE", 0), 5)
        self.assertGreater(s["cases_q2_nonzero_gravity_nonzero"], 10)
        self.assertGreater(s["cases_negative_xprime"], 10)

    def test_two_link_q2_nonzero_gravity_sign_changing_xprime(self):
        par = dict(m1=F(3, 2), m2=1, l1=F(7, 10), c1=F(1, 3), c2=F(2, 5), J1=F(1, 10), J2=F(1, 20), gravity=F(981, 100))
        ch = O.two_link_chain(**par)
        # two C2 spans (third-derivative jump at 1/2), x' = -1+2u style sign change, large q2
        g1 = (F(1, 3), F(1, 2), F(-1, 4), F(1, 5))
        g2 = (F(-1, 2), F(2), F(-3, 2), F(1, 3))
        X = (F(1), F(-3, 2), F(3, 2))
        osps = []
        for left, right in ((F(0), F(1, 2)), (F(1, 2), F(1))):
            q2 = list(g2)
            if left == F(1, 2):  # + 4 (s-1/2)^3
                for n, c in enumerate((-F(1, 8), F(3, 4), -F(3, 2), F(1))):
                    q2[n] += 4 * c
            osps.append(O.OSpan(left, right, (O.shift_poly(g1, left, right - left), O.shift_poly(q2, left, right - left)),
                                O.shift_poly(X, left, right - left)))
        lim = [(-25, 30), (-8, 12)]
        r = certify_torque(cspans(osps), PlanarTwoLink(**par), lim, max_depth=8, max_nodes=2000)
        self.assertEqual(r.status, "CERTIFIED")
        self.assertTrue(replay_torque_leaves(cspans(osps), PlanarTwoLink(**par), lim, r))
        audit(self, ch, osps, lim, r)
        # tighten until violated; witness must be confirmed by the oracle
        tight = [(-25, 30), (-8, F(2))]
        r2 = certify_torque(cspans(osps), PlanarTwoLink(**par), tight, max_depth=8, max_nodes=2000)
        self.assertIn(r2.status, ("VIOLATED_TORQUE", "UNKNOWN_DEPTH", "UNKNOWN_NODES"))
        audit(self, ch, osps, tight, r2)

    def test_interior_true_violation_witness(self):
        ch = O.pendulum_chain(**UNIT_PEND)
        osps = [ospan(0, 1, [[0, 1]], peak_x(1, 16, F(3, 8)))]
        lim = [(-6, F(9, 10))]          # u=0,1/2,1 feasible; true max 1 at u=3/8
        r = certify_torque(cspans(osps), Pendulum(**UNIT_PEND), lim)
        self.assertEqual(r.status, "VIOLATED_TORQUE")
        self.assertNotIn(r.witness["u"], (0, F(1, 2), 1))
        self.assertTrue(O.verify_violation(ch, osps[0], r.witness["u"], 0, lim[0]))
        audit(self, ch, osps, lim, r)

    def test_near_boundary_nondyadic_peak_is_never_certified(self):
        ch = O.pendulum_chain(**UNIT_PEND)
        osps = [ospan(0, 1, [[0, 1]], peak_x(1, 16, F(1, 3)))]
        below = [(-20, 1 - P10)]       # truly violated only within ~1e-15 of u=1/3
        r = certify_torque(cspans(osps), Pendulum(**UNIT_PEND), below, max_depth=12, max_nodes=20000)
        self.assertNotEqual(r.status, "CERTIFIED")
        self.assertIn(r.status, ("UNKNOWN_DEPTH", "UNKNOWN_NODES", "VIOLATED_TORQUE"))
        audit(self, ch, osps, below, r)
        above = [(-20, 1 + P10)]       # truly feasible: must never produce a witness
        r = certify_torque(cspans(osps), Pendulum(**UNIT_PEND), above, max_depth=12, max_nodes=20000)
        self.assertNotEqual(r.status, "VIOLATED_TORQUE")
        audit(self, ch, osps, above, r)

    def test_near_boundary_irrational_gravity_extremum(self):
        O.set_prec(400)
        s1lo, s1hi = O.iv_bounds(O.iv.sin(O.riv(1)))
        O.set_prec(256)
        ch = O.pendulum_chain(1, 1, 1, 1)
        osps = [ospan(0, 1, [[0, 1]], [1])]      # tau = sin(u), max sin(1) at the endpoint
        for hi, must_not in ((s1hi + P10, "VIOLATED_TORQUE"), (s1lo - P10, "CERTIFIED")):
            r = certify_torque(cspans(osps), Pendulum(1, 1, 1, 1), [(-1, hi)], max_depth=10)
            self.assertNotEqual(r.status, must_not)
            audit(self, ch, osps, [(-1, hi)], r)

    def test_torque_exactly_at_limit_and_strict_witness(self):
        osps = [ospan(0, 1, [[0, 0, F(1, 2)]], [1])]   # tau == 1 identically (I=1, g=0)
        ch = O.pendulum_chain(**UNIT_PEND)
        r = certify_torque(cspans(osps), Pendulum(**UNIT_PEND), [(-1, 1)])
        self.assertEqual(r.status, "CERTIFIED")          # closed limit
        audit(self, ch, osps, [(-1, 1)], r)
        r = certify_torque(cspans(osps), Pendulum(**UNIT_PEND), [(-1, 1 - P10)])
        self.assertEqual(r.status, "VIOLATED_TORQUE")
        self.assertGreater(r.witness["torque"][0].lo, 1 - P10)

    def test_asymmetric_limits_lower_side_witness(self):
        ch = O.pendulum_chain(1, 1, 1, 10)
        osps = [ospan(0, 1, [[F(1, 2), F(1, 2)]], [1])]   # tau = 10 sin(q), q in [1/2, 1]
        ok = [(4, 9)]
        r = certify_torque(cspans(osps), Pendulum(1, 1, 1, 10), ok)
        self.assertEqual(r.status, "CERTIFIED")
        audit(self, ch, osps, ok, r)
        bad = [(5, 9)]                                     # 10 sin(1/2) = 4.79 < 5
        r = certify_torque(cspans(osps), Pendulum(1, 1, 1, 10), bad)
        self.assertEqual(r.status, "VIOLATED_TORQUE")
        self.assertLess(r.witness["torque"][0].hi, 5)
        audit(self, ch, osps, bad, r)

    def test_negative_x_cases(self):
        ch = O.pendulum_chain(**UNIT_PEND)
        # tiny negative pocket around non-dyadic 1/3: must never be CERTIFIED
        osps = [ospan(0, 1, [[0, 1]], [F(1, 9) - F(1, 10 ** 12), F(-2, 3), 1])]
        r = certify_torque(cspans(osps), Pendulum(**UNIT_PEND), [(-10, 10)], max_depth=12, max_nodes=20000)
        self.assertNotEqual(r.status, "CERTIFIED")
        audit(self, ch, osps, [(-10, 10)], r)
        # negative only near an interior dyadic point -> VIOLATED_X with a true negative witness
        osps = [ospan(0, 1, [[0, 1]], [F(1, 4) - F(1, 10 ** 12), -1, 1])]
        r = certify_torque(cspans(osps), Pendulum(**UNIT_PEND), [(-10, 10)])
        self.assertEqual(r.status, "VIOLATED_X")
        self.assertLess(O.poly_eval(osps[0].x, r.witness["u"]), 0)

    def test_x_zero_at_interior_points(self):
        ch = O.pendulum_chain(**UNIT_PEND)
        dy = [ospan(0, 1, [[0, 1, F(1, 3)]], [F(1, 4), -1, 1])]       # x=(u-1/2)^2
        r = certify_torque(cspans(dy), Pendulum(**UNIT_PEND), [(-3, 3)])
        self.assertEqual(r.status, "CERTIFIED")
        self.assertIn("REQUIRES_SEPARATE_K3", r.scope)
        audit(self, ch, dy, [(-3, 3)], r)
        nd = [ospan(0, 1, [[0, 1, F(1, 3)]], [F(1, 9), F(-2, 3), 1])]  # x=(u-1/3)^2 >= 0
        r = certify_torque(cspans(nd), Pendulum(**UNIT_PEND), [(-3, 3)], max_depth=10, max_nodes=5000)
        self.assertNotEqual(r.status, "VIOLATED_X")       # would be a false witness
        audit(self, ch, nd, [(-3, 3)], r)

    def test_large_angles_and_fallback_region(self):
        for q, lim in (([[15, 2]], [(-11, 11)]), ([[-40, 80]], [(-11, 11)]), ([[F(16) - F(1, 1000), F(1, 500)]], [(-11, 11)]),
                       ([[200, 1, F(-1, 2)]], [(-15, 15)])):
            ch = O.pendulum_chain(1, 1, 1, 10)
            osps = [ospan(0, 1, q, [1])]
            r = certify_torque(cspans(osps), Pendulum(1, 1, 1, 10), lim, max_depth=8)
            self.assertEqual(r.status, "CERTIFIED", q)
            audit(self, ch, osps, lim, r)
        # constant angle 20 rad: true |tau| = 10|sin 20| = 9.13 <= 9.5, but fallback [-1,1] cannot
        # certify and cannot witness: fail-closed coverage loss (not unsound)
        osps = [ospan(0, 1, [[20]], [1])]
        for lim in ([(-F(95, 10), F(95, 10))], [(-5, 5)]):
            r = certify_torque(cspans(osps), Pendulum(1, 1, 1, 10), lim, max_depth=3)
            self.assertEqual(r.status, "UNKNOWN_DEPTH")

    def test_nonlinear_dependency_two_link_wide_q2(self):
        par = dict(m1=2, m2=3, l1=1, c1=F(1, 2), c2=F(3, 4), J1=F(1, 5), J2=F(1, 4), gravity=F(981, 100))
        ch = O.two_link_chain(**par)
        osps = [ospan(0, 1, [[-1, 3, 0, -1], [-3, 6, 0, 0]], [2, -1, F(1, 2), 0])]
        for lim in ([(-200, 200), (-120, 120)], [(-80, 90), (-40, 50)]):
            r = certify_torque(cspans(osps), PlanarTwoLink(**par), lim, max_depth=9, max_nodes=4000)
            self.assertIn(r.status, ("CERTIFIED", "VIOLATED_TORQUE", "UNKNOWN_DEPTH", "UNKNOWN_NODES"))
            audit(self, ch, osps, lim, r)

    def test_model_parameter_uncertainty_not_representable(self):
        # OBLIGATION GAP: interval/uncertain inertial parameters are rejected at construction.
        with self.assertRaises(TypeError):
            PlanarTwoLink(I(1, 2), 1, 1, 1, 1, 0, 0, 0)

    def test_regularity_at_knots_uses_global_derivatives(self):
        P = Pendulum(1, 1, 1, 1)
        # local slopes equal (1) but global slopes 3 vs 3/2 -> not C1 in s -> must be rejected
        a = PathSpan.from_values(0, F(1, 3), [[0, 1]], [1])
        b = PathSpan.from_values(F(1, 3), 1, [[1, 1]], [1])
        self.assertEqual(certify_torque([a, b], P, [(-9, 9)]).status, "INVALID_PATH")
        # a genuinely C2 path with unequal knots is accepted
        g = (F(1, 5), F(1, 2), F(-1, 3), F(1, 7))
        sps = [PathSpan(l, r, (O.shift_poly(g, l, r - l),), (2,)) for l, r in ((F(0), F(1, 3)), (F(1, 3), F(1)))]
        self.assertNotEqual(certify_torque(sps, P, [(-9, 9)]).status, "INVALID_PATH")
        # q'' jump (C1 only) rejected; x' jump rejected; x'' jump allowed (C1 x)
        bad_q = [sps[0], replace(sps[1], q=((sps[1].q[0][0], sps[1].q[0][1], sps[1].q[0][2] + 1, sps[1].q[0][3]),))]
        self.assertEqual(certify_torque(bad_q, P, [(-9, 9)]).status, "INVALID_PATH")
        bad_x = [replace(sps[0], x=(2, 0)), replace(sps[1], x=(2, F(1, 10)))]
        self.assertEqual(certify_torque(bad_x, P, [(-9, 9)]).status, "INVALID_PATH")
        ok_x = [replace(sps[0], x=(2,)), replace(sps[1], x=(2, 0, F(1, 10)))]
        self.assertNotEqual(certify_torque(ok_x, P, [(-9, 9)]).status, "INVALID_PATH")
        gap = [sps[0], replace(sps[1], left=F(1, 2))]
        self.assertEqual(certify_torque(gap, P, [(-9, 9)]).status, "INVALID_PATH")

    def test_budget_fail_closed(self):
        P = Pendulum(1, 1, 1, 1)
        sp = [PathSpan.from_values(0, 1, [[0, 1]], [1])]
        for mn in (1, 2, 3, 4):   # this case needs exactly 5 nodes
            r = certify_torque(sp, P, [(-F(9, 10), F(9, 10))], max_nodes=mn)
            self.assertIn(r.status, ("UNKNOWN_NODES", "UNKNOWN_DEPTH"))
            self.assertLessEqual(r.nodes, mn)
        self.assertEqual(certify_torque(sp, P, [(-F(9, 10), F(9, 10))], max_depth=0).status, "UNKNOWN_DEPTH")
        for kw in ({"max_depth": -1}, {"max_depth": 2.0}, {"max_nodes": 0}, {"max_nodes": "9"}):
            self.assertEqual(certify_torque(sp, P, [(-1, 1)], **kw).status, "INVALID_INPUT")
        self.assertEqual(certify_torque(sp, P, [(-1, 1), (-1, 1)]).status, "INVALID_INPUT")
        self.assertEqual(certify_torque(sp, P, [(1, 1)]).status, "INVALID_INPUT")
        self.assertEqual(certify_torque([], P, [(-1, 1)]).status, "INVALID_PATH")
        # dof field is a public dataclass field: a mismatched dof must fail closed, not certify
        self.assertEqual(certify_torque(sp, PlanarTwoLink(1, 1, 1, 1, 1, 0, 0, 0, dof=1), [(-9, 9)]).status,
                         "INVALID_INPUT")
        two = [PathSpan.from_values(0, 1, [[0], [0]], [1])]
        self.assertEqual(certify_torque(two, Pendulum(1, 1, 1, 1, dof=2), [(-9, 9)] * 2).status, "INVALID_INPUT")
        # NOTE: bool budgets are accepted as ints (max_depth=True == 1) - cosmetic.
        self.assertEqual(certify_torque(sp, P, [(-1, 1)], max_depth=True).status, "CERTIFIED")


@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class ReplayAttacks(unittest.TestCase):
    def setUp(self):
        self.par = dict(m1=1, m2=2, l1=1, c1=F(1, 2), c2=F(1, 2), J1=F(1, 10), J2=F(1, 10), gravity=F(981, 100))
        self.model = PlanarTwoLink(**self.par)
        self.chain = O.two_link_chain(**self.par)
        g = [(F(1, 4), F(1, 2), F(-1, 4), 0), (F(1, 3), F(-1), F(1, 2), F(1, 5))]
        self.osps = [O.OSpan(l, r, tuple(O.shift_poly(p, l, r - l) for p in g), O.shift_poly((2, -1, F(1, 2)), l, r - l))
                     for l, r in ((F(0), F(1, 2)), (F(1, 2), F(1)))]
        self.spans = cspans(self.osps)
        self.lim = [(29, 32), (7, 10)]          # asymmetric, both positive; oracle range ~[29.76,31.52],[7.92,9.40]
        self.r = certify_torque(self.spans, self.model, self.lim, max_depth=10, max_nodes=4000)
        self.assertEqual(self.r.status, "CERTIFIED")
        self.assertGreater(len(self.r.leaves), 3)
        self.assertTrue(replay_torque_leaves(self.spans, self.model, self.lim, self.r))

    def rp(self, res=None, model=None, lim=None, spans=None):
        return replay_torque_leaves(spans or self.spans, model or self.model, lim or self.lim, res or self.r)

    def test_omitted_duplicate_overlapping_and_foreign_leaves(self):
        L = self.r.leaves
        self.assertFalse(self.rp(replace(self.r, leaves=L[1:])))
        self.assertFalse(self.rp(replace(self.r, leaves=L + (L[0],))))
        parent = replace(L[0], path=L[0].path[:-1]) if L[0].path else None
        if parent is not None:
            self.assertFalse(self.rp(replace(self.r, leaves=L + (parent,))))
        self.assertFalse(self.rp(replace(self.r, leaves=L + (replace(L[0], span=2),))))
        self.assertFalse(self.rp(replace(self.r, leaves=L + (replace(L[0], span=-1),))))
        self.assertFalse(self.rp(replace(self.r, leaves=(replace(L[0], path=L[0].path + "X"),) + L[1:])))
        # merge two siblings into a parent with a forged (narrow) box
        sib = [lf for lf in L if lf.span == 0]
        self.assertFalse(self.rp(replace(self.r, leaves=tuple(lf for lf in L if lf.span != 0) +
                                         (replace(sib[0], path=""),))))

    def test_misordered_leaves_are_accepted_and_still_sound(self):
        L = list(self.r.leaves)
        random.Random(4).shuffle(L)
        res = replace(self.r, leaves=tuple(L))
        self.assertTrue(self.rp(res))
        audit(self, self.chain, self.osps, self.lim, res)

    def test_too_narrow_boxes_rejected(self):
        for k, lf in enumerate(self.r.leaves[:6]):
            for j in range(2):
                t = list(lf.torque)
                t[j] = I(t[j].lo + P10, t[j].hi) if k % 2 else I(t[j].lo, t[j].hi - P10)
                L = list(self.r.leaves)
                L[k] = replace(lf, torque=tuple(t))
                self.assertFalse(self.rp(replace(self.r, leaves=tuple(L))))

    def test_altered_model_parameters(self):
        rejected = 0
        rng = random.Random(15)
        for key in self.par:
            for fac in (F(3, 2), F(1, 2), F(11, 10)):
                p = dict(self.par)
                p[key] = p[key] * fac if p[key] else F(1, 5)
                m = PlanarTwoLink(**p)
                ok = self.rp(model=m)
                rejected += not ok
                if ok:   # acceptance must still be a TRUE certificate for the altered model
                    C_ = C.audit_result(rng, O.two_link_chain(**p), self.osps, self.lim, self.r, dense=48)
                    self.assertEqual(C_["disjoint"], [])
        self.assertGreater(rejected, 10)
        self.assertFalse(self.rp(model=Pendulum(1, 1, 1, 1)))

    def test_status_limits_and_fake_witness(self):
        self.assertFalse(self.rp(replace(self.r, status="VIOLATED_TORQUE")))
        self.assertFalse(self.rp(lim=[(29, 31), (7, 10)]))
        # a CERTIFIED result carrying a fabricated witness is still judged on its leaves only
        self.assertTrue(self.rp(replace(self.r, witness={"span": 0, "u": F(1, 2), "joint": 0, "torque": (I(99, 100),)})))
        # GAP: there is no witness replay in the reviewed code. A forged VIOLATED result is
        # simply "not CERTIFIED" for replay; the independent oracle rejects the fake witness.
        fake = K4Result("VIOLATED_TORQUE", R.SCOPE, 1, (), {"span": 0, "u": F(1, 4), "joint": 0,
                                                            "torque": (I(100, 101), I(0, 0))}, "forged")
        self.assertFalse(self.rp(fake))
        self.assertFalse(O.verify_violation(self.chain, self.osps[0], F(1, 4), 0, self.lim[0]))

    def test_stored_boxes_wider_than_limits_are_accepted_harmlessly(self):
        # Replay checks the RECOMPUTED box against limits, not the stored one. A stored box
        # outside the limits is accepted (cosmetic; the verdict never relies on stored boxes).
        lf = self.r.leaves[0]
        wide = replace(lf, torque=(I(-1000, 1000), lf.torque[1]))
        self.assertTrue(self.rp(replace(self.r, leaves=(wide,) + self.r.leaves[1:])))

    def test_replay_rejects_spans_swapped(self):
        self.assertFalse(self.rp(spans=self.spans[::-1]))   # not contiguous -> regularity fails


@unittest.skipUnless(HAVE_MPMATH, "mpmath required")
class AffineAttacks(unittest.TestCase):
    def setUp(self):
        self.par = dict(m1=F(3, 2), m2=1, l1=F(7, 10), c1=F(1, 3), c2=F(2, 5), J1=F(1, 10), J2=F(1, 20), gravity=F(981, 100))
        self.model = PlanarTwoLink(**self.par)
        self.chain = O.two_link_chain(**self.par)
        self.q = ((F(1, 3), F(1, 2), F(-1, 4), F(1, 5)), (F(-1, 2), 2, F(-3, 2), F(1, 3)))
        self.os = [O.OSpan(F(0), F(1), self.q, (F(3), F(-2), F(1, 2)))]
        self.sp = cspans(self.os)
        self.box = AF.prepare_geometry(self.sp, self.model, depth=4)

    def test_prepared_ABG_boxes_contain_oracle(self):
        bad = []
        rng = random.Random(16)
        for lf in self.box.leaves:
            a, b = O.path_interval(lf.path)
            for u in C.leaf_samples(rng, a, b, 2):
                q, qp, qpp, _, _ = O.path_state(self.os[0], u)
                for k, rec in enumerate((lf.A, lf.B, lf.G)):
                    for j in range(2):
                        v = O.check_box(rec[j].lo, rec[j].hi, lambda k=k, j=j: O.abg(self.chain, q, qp, qpp)[k][j])
                        if v not in ("CONTAINED", "POINT_CONSISTENT"):
                            bad.append((lf.path, str(u), k, j, v))
        self.assertEqual(bad, [])

    def test_signed_xprime_alternative_timings(self):
        for x in ((F(3), F(-2), F(1, 2)), (F(1, 2), F(3), F(-1)), (F(4), F(-6), F(3)), (F(2), F(1), F(-2), F(1, 2))):
            alt = [O.OSpan(F(0), F(1), self.q, x)]
            lim = [(-60, 60), (-25, 25)]
            r = AF.certify_with_prepared_geometry(cspans(alt), self.model, lim, self.box)
            self.assertIn(r.status, ("CERTIFIED", "UNKNOWN_PREPARED_ENCLOSURE"))
            if r.status == "CERTIFIED":
                self.assertTrue(replay_torque_leaves(cspans(alt), self.model, lim, r))
            audit(self, self.chain, alt, lim, r)

    def test_negative_x_is_not_certified_via_prepared(self):
        alt = cspans([O.OSpan(F(0), F(1), self.q, (F(1, 9) - F(1, 10 ** 9), F(-2, 3), F(1)))])
        r = AF.certify_with_prepared_geometry(alt, self.model, [(-10 ** 6, 10 ** 6)] * 2, self.box)
        self.assertNotEqual(r.status, "CERTIFIED")

    def test_model_mutation_and_forged_prepared(self):
        m = PlanarTwoLink(**self.par)
        box = AF.prepare_geometry(self.sp, m, depth=2)
        object.__setattr__(m, "m2", F(5))           # bypass frozen dataclass
        self.assertFalse(AF.verify_prepared_geometry(self.sp, m, box))
        self.assertEqual(AF.certify_with_prepared_geometry(self.sp, m, [(-99, 99)] * 2, box).status,
                         "INVALID_PREPARED_GEOMETRY")
        # forged: correct digests of the target geometry, but boxes from another geometry
        other = [replace(self.sp[0], q=(self.sp[0].q[0], tuple(c * 2 for c in self.sp[0].q[1])))]
        ob = AF.prepare_geometry(other, self.model, depth=2)
        forged = replace(ob, geometry_sha256=AF.geometry_digest(self.sp))
        self.assertFalse(AF.verify_prepared_geometry(self.sp, self.model, forged))
        # duplicated + missing leaf with correct count
        L = self.box.leaves
        self.assertFalse(AF.verify_prepared_geometry(self.sp, self.model, replace(self.box, leaves=(L[0],) + L[:-1])))
        # bogus span index / path char / depth
        self.assertFalse(AF.verify_prepared_geometry(self.sp, self.model,
                                                     replace(self.box, leaves=(replace(L[0], span=1),) + L[1:])))
        self.assertFalse(AF.verify_prepared_geometry(self.sp, self.model,
                                                     replace(self.box, leaves=(replace(L[0], path="LLLX"),) + L[1:])))
        self.assertFalse(AF.verify_prepared_geometry(self.sp, self.model, replace(self.box, depth=3)))
        # narrower recorded box
        t = L[0].A[0]
        self.assertFalse(AF.verify_prepared_geometry(self.sp, self.model,
                                                     replace(self.box, leaves=(replace(L[0], A=(I(t.lo + P10, t.hi), L[0].A[1])),) + L[1:])))

    def test_digest_semantics(self):
        # digest excludes x (intended), includes knots; semantically equal q with an explicit
        # trailing zero coefficient hashes differently -> fail-closed (spurious INVALID), not unsound.
        d = AF.geometry_digest(self.sp)
        self.assertEqual(d, AF.geometry_digest([replace(self.sp[0], x=(1,))]))
        self.assertNotEqual(d, AF.geometry_digest([replace(self.sp[0], right=2)]))
        padded = [PathSpan(0, 1, ((0, 1), (0,)), (1,))]
        plain = [PathSpan(0, 1, ((0, 1, 0), (0,)), (1,))]
        self.assertNotEqual(AF.geometry_digest(padded), AF.geometry_digest(plain))
        self.assertNotEqual(AF.model_digest(self.model), AF.model_digest(PlanarTwoLink(**{**self.par, "J2": F(1, 21)})))


if __name__ == "__main__":
    unittest.main()
