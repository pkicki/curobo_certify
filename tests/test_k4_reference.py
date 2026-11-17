"""Synthetic analytic K4 reference tests; NOT Panda or production native parity."""
import math
import random
import unittest
from dataclasses import replace
from fractions import Fraction as F

from curobo_certify.k4_reference import (
    I, Pendulum, PlanarTwoLink, PathSpan, K4Result, certify_torque,
    replay_torque_leaves, isin, icos, _point_torque, _torque_bounds,
    eval_poly, split, bernstein,
)


class K4ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.pendulum = Pendulum(1, 1, 1, 1)
        self.path = PathSpan.from_values(0, 1, [[0, 1]], [1])

    def test_exact_zero_angle(self):
        self.assertEqual(isin(I.point(0)), I.point(0))
        self.assertEqual(icos(I.point(0)), I.point(1))

    def test_rational_interval_arithmetic(self):
        a = I(F(-2), F(3)); b = I(F(-5), F(7))
        self.assertEqual(a * b, I(F(-15), F(21)))
        self.assertEqual(a + b, I(F(-7), F(10)))
        self.assertEqual(a / I(F(2), F(4)), I(F(-1), F(3, 2)))
        with self.assertRaises(ZeroDivisionError):
            _ = a / I(F(-1), F(1))

    def test_sin_cos_enclose_samples(self):
        rng = random.Random(20261117)
        for _ in range(300):
            center = F(rng.randint(-600, 600), 100)
            radius = F(rng.randint(0, 200), 1000)
            x = I(center - radius, center + radius)
            for f, func in ((isin, math.sin), (icos, math.cos)):
                bound = f(x)
                for j in range(11):
                    v = func(float(x.lo + (x.hi-x.lo)*j/10))
                    self.assertLessEqual(float(bound.lo)-1e-14, v)
                    self.assertLessEqual(v, float(bound.hi)+1e-14)
        self.assertEqual(isin(I(20, 21)), I(F(-1), F(1)))

    def test_constant_zero_torque_certifies_and_replays(self):
        r = certify_torque([PathSpan.from_values(0, 1, [[0]], [1])],
                           self.pendulum, [(-1, 1)])
        self.assertEqual(r.status, "CERTIFIED")
        self.assertTrue(replay_torque_leaves([PathSpan.from_values(0, 1, [[0]], [1])],
                                              self.pendulum, [(-1, 1)], r))

    def test_bisection_helps_without_modifying_path(self):
        no_split = certify_torque([self.path], self.pendulum, [(-F(9,10), F(9,10))], max_depth=0)
        self.assertEqual(no_split.status, "UNKNOWN_DEPTH")
        r = certify_torque([self.path], self.pendulum, [(-F(9,10), F(9,10))], max_depth=12)
        self.assertEqual(r.status, "CERTIFIED")
        self.assertGreater(len(r.leaves), 1)
        self.assertTrue(replay_torque_leaves([self.path], self.pendulum,
                                              [(-F(9,10), F(9,10))], r))

    def test_genuine_torque_violation_witness(self):
        r = certify_torque([self.path], self.pendulum, [(-F(1,2), F(1,2))])
        self.assertEqual(r.status, "VIOLATED_TORQUE")
        self.assertEqual(r.witness["u"], 1)
        self.assertGreater(r.witness["torque"][0].lo, F(1,2))

    def test_x_negative_is_not_certified(self):
        s = PathSpan.from_values(0, 1, [[0]], [-1])
        r = certify_torque([s], self.pendulum, [(-1, 1)])
        self.assertEqual(r.status, "VIOLATED_X")

    def test_x_zero_interior_requires_separate_k3(self):
        s = PathSpan.from_values(0, 1, [[0]], [F(1,4), -1, 1])
        r = certify_torque([s], self.pendulum, [(-1, 1)])
        self.assertEqual(r.status, "CERTIFIED")
        self.assertIn("REQUIRES_SEPARATE_K3", r.scope)

    def test_invalid_regularity_and_limits_fail_closed(self):
        a = PathSpan.from_values(0, F(1,2), [[0]], [1])
        b = PathSpan.from_values(F(1,2), 1, [[1]], [1])
        self.assertEqual(certify_torque([a, b], self.pendulum, [(-1, 1)]).status, "INVALID_PATH")
        self.assertEqual(certify_torque([self.path], self.pendulum, [(1, -1)]).status, "INVALID_INPUT")
        self.assertEqual(certify_torque([self.path], self.pendulum, [(-1, 1)], max_nodes=0).status, "INVALID_INPUT")

    def test_replay_rejects_tampering(self):
        r = certify_torque([self.path], self.pendulum, [(-1, 1)])
        self.assertTrue(replay_torque_leaves([self.path], self.pendulum, [(-1, 1)], r))
        self.assertFalse(replay_torque_leaves([self.path], self.pendulum, [(-1, 1)],
                                              replace(r, leaves=())))
        self.assertFalse(replay_torque_leaves([self.path], self.pendulum, [(-1, 1)],
                                              replace(r, leaves=r.leaves+r.leaves)))
        self.assertFalse(replay_torque_leaves([self.path], self.pendulum, [(-1, 1)],
                                              replace(r, leaves=(replace(r.leaves[0],torque=(I.point(0),)),))))
        self.assertFalse(replay_torque_leaves([self.path], self.pendulum, [(-F(1,2),F(1,2))],r))

    def test_planar_two_link_exact_inertial_case(self):
        m = PlanarTwoLink(1,1,1,F(1,2),F(1,2),0,0,0)
        p = PathSpan.from_values(0,1,[[0,0,F(1,2)],[0]],[1])
        exact = _point_torque(p,m,F(3,5))
        self.assertEqual(exact, (I.point(F(5,2)), I.point(F(3,4))))
        r = certify_torque([p],m,[(-3,3),(-1,1)])
        self.assertEqual(r.status,"CERTIFIED")
        self.assertTrue(replay_torque_leaves([p],m,[(-3,3),(-1,1)],r))
        self.assertEqual(certify_torque([p],m,[(-2,2),(-1,1)]).status,"VIOLATED_TORQUE")

    def test_affinity_in_x_xprime_at_fixed_s(self):
        m = PlanarTwoLink(1,2,1,F(1,2),F(1,3),F(1,10),F(1,20),F(981,100))
        q = (I.point(F(1,4)),I.point(F(-1,3)))
        d = (I.point(F(2,5)),I.point(F(1,7)))
        dd = (I.point(F(-1,9)),I.point(F(1,11)))
        A,B,G=m.coefficients(q,d,dd)
        for x,xp in ((F(1,2),F(1,5)),(F(11,4),F(-7,10))):
            lhs=_torque_bounds(m,q,d,dd,I.point(x),I.point(xp))
            rhs=tuple(G[j]+A[j]*x+B[j]*xp for j in range(2))
            self.assertEqual(lhs,rhs)

    def test_bernstein_bounds_rational_quadratic(self):
        b = bernstein((F(1,4), -1, 1))
        left,right=split(b)
        self.assertEqual(left[-1],F(0))
        self.assertEqual(right[0],F(0))


if __name__ == "__main__":
    unittest.main()
