from fractions import Fraction
import sys
import random
import unittest
import json
import hashlib
import pickle
from pathlib import Path

sys.path.insert(0, "src")

from curobo_certify.boundary_reference import Boundaries, Span
from curobo_certify.exact_k3_reference import certify_k3
from curobo_certify.piecewise_cpu import certify_piecewise_k3, replay_piecewise_k3


class PiecewiseCPUReferenceTests(unittest.TestCase):
    def test_two_span_analytic_fixture_passes_both_backends_and_replay(self):
        raw = Path("fixtures/boundary_multispan_analytic.json").read_bytes()
        data = json.loads(raw)
        spans = [Span.from_values(s["left"], s["right"], s["q"], s["x"])
                 for s in data["spans"]]
        targets = Boundaries.from_values(**data["targets"])
        limits = (data["limits"]["velocity"], data["limits"]["acceleration"],
                  data["limits"]["jerk"])
        outward = certify_piecewise_k3(spans, targets, *limits)
        exact = certify_k3(spans, targets, *limits)
        self.assertEqual(outward.k3_status, "CERTIFIED_K3")
        self.assertEqual(exact.status, "CERTIFIED_K3_REFERENCE")
        self.assertEqual(replay_piecewise_k3(spans, targets, *limits, outward)[0], True)

    def test_exact_oracle_and_outward_checker_agree_on_256_seeded_cases(self):
        rng = random.Random(20261110)
        for case in range(256):
            denom = rng.choice((3, 5, 7, 11, 16, 31))
            q = tuple(Fraction(rng.randint(-3, 3), denom) for _ in range(4))
            # A positive constant squared speed makes the traversal regular;
            # deliberately vary exact rational span lengths to exercise h^-k.
            x = (Fraction(rng.randint(1, 4), 4),)
            right = Fraction(rng.choice((1, 2, 3, 5)), rng.choice((2, 3, 5, 7)))
            span = Span.from_values(0, right, [q], x)
            targets = Boundaries.from_values(
                q_start=[q[0]],
                q_end=[sum(q, Fraction(0))],
                x_start=x[0],
                x_end=x[0],
            )
            limits = ((1_000_000,), (1_000_000,), (1_000_000,))
            exact_result = certify_k3([span], targets, *limits, max_depth=8, max_nodes=1000)
            interval_result = certify_piecewise_k3(
                [span], targets, *limits, max_depth=8, max_nodes=1000
            )
            self.assertEqual(exact_result.status, "CERTIFIED_K3_REFERENCE", f"case {case}")
            self.assertEqual(interval_result.k3_status, "CERTIFIED_K3", f"case {case}")
            self.assertEqual(interval_result.full_contract_status, "UNSUPPORTED_DYNAMICS")
            self.assertEqual(replay_piecewise_k3([span], targets, *limits, interval_result),
                             (True, "all exact boundaries and outward-rounded K3 leaves replayed"))

    def test_strict_rest_and_endpoint_integrability_are_in_conjunction(self):
        # Linear endpoint zeros of x give finite ds/sqrt(x); smoothstep q
        # makes both strict-rest accelerations exactly zero.
        span = Span.from_values(0, 1, [[0, 0, 3, -2]], [0, 1, -1])
        target = Boundaries.from_values(
            q_start=[0], q_end=[1], x_start=0, x_end=0,
            strict_rest=True,
        )
        result = certify_piecewise_k3([span], target, (10,), (10,), (10,))
        self.assertEqual(result.k3_status, "CERTIFIED_K3")
        self.assertEqual(replay_piecewise_k3([span], target, (10,), (10,), (10,), result)[0], True)

    def test_nonintegrable_endpoint_fails_before_residual_proof(self):
        span = Span.from_values(0, 1, [[0, 1]], [0, 0, 1])
        target = Boundaries.from_values(q_start=[0], q_end=[1], x_start=0, x_end=1)
        result = certify_piecewise_k3([span], target, (10,), (10,), (10,))
        self.assertEqual(result.k3_status, "VIOLATED")
        self.assertEqual(result.boundary_status, "VIOLATED")
        self.assertEqual(result.span_results, [])

    def test_knot_discontinuity_blocks_piecewise_certificate(self):
        spans = [
            Span.from_values(0, Fraction(1, 2), [[0, Fraction(1, 2)]], [1]),
            Span.from_values(Fraction(1, 2), 1, [[Fraction(3, 4), Fraction(1, 2)]], [1]),
        ]
        targets = Boundaries.from_values(q_start=[0], q_end=[Fraction(5, 4)], x_start=1, x_end=1)
        result = certify_piecewise_k3(spans, targets, (10,), (10,), (10,))
        self.assertEqual(result.k3_status, "VIOLATED")
        self.assertEqual(result.boundary_status, "VIOLATED")

    def test_exact_and_outward_oracles_agree_on_hidden_velocity_violation(self):
        span = Span.from_values(0, 1, [[0, 1]], [1, 14, -14])
        targets = Boundaries.from_values(q_start=[0], q_end=[1], x_start=1, x_end=1)
        limits = ((1,), (100,), (100,))
        exact = certify_k3([span], targets, *limits)
        outward = certify_piecewise_k3([span], targets, *limits)
        self.assertEqual(exact.status, "VIOLATED")
        self.assertEqual(outward.k3_status, "VIOLATED")

    def test_proof_replay_rejects_a_tampered_leaf(self):
        span = Span.from_values(0, 1, [[0, 1]], [1])
        targets = Boundaries.from_values(q_start=[0], q_end=[1], x_start=1, x_end=1)
        result = certify_piecewise_k3([span], targets, (2,), (2,), (2,))
        self.assertEqual(result.k3_status, "CERTIFIED_K3")
        result.span_results[0][0].leaves[0].path = "0"
        self.assertFalse(replay_piecewise_k3([span], targets, (2,), (2,), (2,), result)[0])

    def test_near_zero_but_positive_speed_remains_certifiable(self):
        tiny = Fraction(1, 10**300)
        span = Span.from_values(0, 1, [[0, 1]], [tiny])
        targets = Boundaries.from_values(q_start=[0], q_end=[1], x_start=tiny, x_end=tiny)
        result = certify_piecewise_k3(
            [span], targets, (Fraction(1, 10**100),), (1,), (1,)
        )
        self.assertEqual(result.k3_status, "CERTIFIED_K3")

    def test_tiny_global_span_scales_derivatives_without_float_truncation(self):
        right = Fraction(1, 10**50)
        span = Span.from_values(0, right, [[0, 1]], [1])
        targets = Boundaries.from_values(q_start=[0], q_end=[1], x_start=1, x_end=1)
        result = certify_piecewise_k3([span], targets, (Fraction(10**51),), (1,), (1,))
        self.assertEqual(result.k3_status, "CERTIFIED_K3")

    def test_inputs_are_immutable_across_certification_and_replay(self):
        spans = [Span.from_values(0, 1, [[0, 1]], [1])]
        targets = Boundaries.from_values(q_start=[0], q_end=[1], x_start=1, x_end=1)
        before = hashlib.sha256(pickle.dumps((spans, targets))).hexdigest()
        result = certify_piecewise_k3(spans, targets, (2,), (2,), (2,))
        self.assertTrue(replay_piecewise_k3(spans, targets, (2,), (2,), (2,), result)[0])
        after = hashlib.sha256(pickle.dumps((spans, targets))).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
