import sys
import random
import unittest
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from curobo_certify.boundary_reference import Span, Boundaries
from curobo_certify.exact_k3_reference import (bernstein_coeffs, midpoint_split,
    certify_k3, replay_k3_leaves, build_residuals)


def case(x=(1,), q=(0, 1), x_start=1, x_end=1):
    span = Span.from_values(0, 1, [q], x)
    limits = (3,), (10,), (50,)
    targets = Boundaries.from_values([sum(q[:1])], [sum(q)], x_start, x_end)
    return (span,), targets, limits


class ExactK3Tests(unittest.TestCase):
    def test_seeded_256_exact_sample_diagnostics(self):
        rng = random.Random(20261110)
        for _ in range(256):
            qa = (0, rng.randrange(-3, 4), rng.randrange(-2, 3), rng.randrange(-1, 2))
            xa = (1, rng.randrange(-2, 3), rng.randrange(-1, 2))
            spans, targets, _ = case(xa, qa, xa[0], sum(xa))
            r = certify_k3(spans, targets, [5], [20], [100], max_depth=6, max_nodes=2048)
            if r.status == 'CERTIFIED_K3_REFERENCE':
                self.assertTrue(replay_k3_leaves(spans, targets, [5], [20], [100], r))
                for _, _, residual in build_residuals(spans[0], [5], [20], [100]):
                    for step in range(257):
                        u = F(step, 256)
                        val = sum((a * u**i for i, a in enumerate(residual)), F(0))
                        self.assertLessEqual(val, 0)

    def test_bernstein_conversion_and_exact_split(self):
        b = bernstein_coeffs((F(-1), F(24, 5), F(-24, 5)))
        self.assertEqual(b, (F(-1), F(7, 5), F(-1)))
        left, right = midpoint_split(b)
        self.assertEqual(left[0], b[0])
        self.assertEqual(right[-1], b[-1])
        self.assertEqual(left[-1], right[0])

    def test_frozen_q_linear_constant_x_certifies(self):
        spans, targets, limits = case()
        result = certify_k3(spans, targets, *limits)
        self.assertEqual(result.status, 'CERTIFIED_K3_REFERENCE', result.detail)
        self.assertTrue(replay_k3_leaves(spans, targets, *limits, result))
        self.assertEqual(len(result.proof_leaves), 5)

    def test_velocity_limit_rigorous_violation(self):
        spans, targets, _ = case()
        r = certify_k3(spans, targets, [F(1, 2)], [10], [50])
        self.assertEqual(r.status, 'VIOLATED')
        self.assertIn('velocity', r.violation)

    def test_acceleration_violation(self):
        spans, targets, _ = case((1, 1), (0, 1), 1, 2)
        r = certify_k3(spans, targets, [5], [F(1, 4)], [50])
        self.assertEqual(r.status, 'VIOLATED')
        self.assertIn('acceleration', r.violation)

    def test_jerk_violation(self):
        spans, targets, _ = case((1, 1), (0, 0, 1), 1, 2)
        r = certify_k3(spans, targets, [5], [10], [F(1, 2)])
        self.assertEqual(r.status, 'VIOLATED')
        self.assertIn('jerk', r.violation)

    def test_boundary_unknown_blocks_full_proof(self):
        spans, _, limits = case()
        r = certify_k3(spans, Boundaries(), *limits)
        self.assertEqual(r.status, 'UNKNOWN_BOUNDARY')

    def test_nonintegrable_zero(self):
        spans, targets, limits = case((0, 0, 1), (0, 1), 0, 1)
        self.assertEqual(certify_k3(spans, targets, *limits).status, 'VIOLATED')

    def test_negative_x(self):
        spans, targets, limits = case((1, -5, 5), (0, 1))
        self.assertEqual(certify_k3(spans, targets, *limits, max_depth=10).status, 'VIOLATED')

    def test_strict_positivity_unknown_interior_root(self):
        # x=(u-1/2)^2, has interior stationary knot/root.
        spans, targets, _ = case((F(1, 4), -1, 1), (0, 1), F(1, 4), F(1, 4))
        r = certify_k3(spans, targets, [3], [10], [50])
        self.assertEqual(r.status, 'UNKNOWN_STATIONARY')

    def test_zero_speed_endpoint_linear_integrable(self):
        spans, targets, _ = case((0, 1), (0, 1), 0, 1)
        r = certify_k3(spans, targets, [3], [10], [50])
        self.assertEqual(r.status, 'CERTIFIED_K3_REFERENCE', r.detail)
        self.assertTrue(replay_k3_leaves(spans, targets, [3], [10], [50], r))

    def test_unsplit_hull_fails_but_two_children_certify(self):
        # g(u)=-1+4.8u-4.8u^2 has positive Bernstein b_1 despite g<= -0.0?
        # Use a known negative residual with hull inconclusive, g=-1+3u-3u².
        b = bernstein_coeffs((F(-1), F(3), F(-3)))
        self.assertTrue(any(v > 0 for v in b))
        self.assertTrue(all(v <= 0 for v in midpoint_split(b)[0]))
        self.assertTrue(all(v <= 0 for v in midpoint_split(b)[1]))

    def test_proof_replay_tamper_rejected(self):
        spans, targets, limits = case()
        r = certify_k3(spans, targets, *limits)
        self.assertTrue(r.proof_leaves)
        bad_leaf = replace(r.proof_leaves[0], coefficients=('0',))
        bad = replace(r, proof_leaves=(bad_leaf, *r.proof_leaves[1:]))
        self.assertFalse(replay_k3_leaves(spans, targets, *limits, bad))
        missing = replace(r, proof_leaves=r.proof_leaves[1:])
        self.assertFalse(replay_k3_leaves(spans, targets, *limits, missing))

    def test_bad_limits_fail_closed(self):
        spans, targets, _ = case()
        self.assertEqual(certify_k3(spans, targets, [0], [10], [50]).status, 'INVALID_INPUT')

    def test_nodes_budget_terminates(self):
        spans, targets, _ = case()
        r = certify_k3(spans, targets, [3], [10], [50], max_nodes=1)
        self.assertEqual(r.status, 'UNKNOWN_NODES')

    def test_interval_derivative_global_scaling(self):
        span = Span.from_values(0, F(1, 4), [[0, 0, 1]], [1])
        data = list(build_residuals(span, [100], [1000], [1000]))
        # q_s=8u, v²=64u²; exact Bernstein coefficient of u² term.
        self.assertEqual(data[1][2][-1], F(64, 10000))


if __name__ == '__main__':
    unittest.main()
