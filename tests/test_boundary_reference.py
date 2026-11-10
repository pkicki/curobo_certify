"""Independent exact arithmetic contract tests; not native TimeOpt parity."""
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curobo_certify.boundary_reference import (Span, Boundaries, check_boundaries,
                                                rational, derivative)


def valid_pair():
    # q(s)=s^2, x(s)=s(2-s), represented on s in [0,1] with a knot at 1/2.
    a = Span.from_values(0, "1/2", [[0, 0, "1/4"]], [0, 1, "-1/4"])
    # x(1/2 + u/2) = 3/4 + (u/2) - u^2/4
    b = Span.from_values("1/2", 1, [["1/4", "1/2", "1/4"]],
                         ["3/4", "1/2", "-1/4"])
    return a, b


class BoundaryReferenceTests(unittest.TestCase):
    def test_valid_free_endpoint(self):
        a, b = valid_pair()
        result = check_boundaries([a, b], Boundaries.from_values([0], [1], 0, 1))
        self.assertEqual(result.status, "PASS", result.checks)

    def test_missing_target_fails_closed(self):
        self.assertEqual(check_boundaries(valid_pair(), Boundaries()).status, "UNKNOWN")

    def test_strict_rest_does_not_accept_nonzero_acceleration(self):
        a, b = valid_pair()
        targets = Boundaries.from_values([0], [1], 0, 1, strict_rest=True)
        self.assertEqual(check_boundaries([a, b], targets).status, "VIOLATED")

    def test_real_strict_rest(self):
        # q(s)=s, x=6 s(1-s); at endpoints acceleration !=0, hence NOT strict rest.
        span = Span.from_values(0, 1, [[0, 1]], [0, 6, -6])
        bad = Boundaries.from_values([0], [1], 0, 0, strict_rest=True)
        self.assertEqual(check_boundaries([span], bad).status, "VIOLATED")
        # x=s^2 (1-s)^2 makes end acceleration zero, but traversal time infinite.
        # Degree 4 exceeds the declared cubic contract; a zero function is unknown.
        still = Span.from_values(0, 1, [[0, 1]], [0])
        self.assertEqual(check_boundaries([still], bad).status, "UNKNOWN")

    def test_knot_position_jump_detected(self):
        a, b = valid_pair()
        b = Span.from_values(b.left, b.right, [["1/3", "1/2", "1/4"]], b.x)
        self.assertEqual(check_boundaries([a, b], Boundaries.from_values([0], [1], 0, 1)).status, "VIOLATED")

    def test_knot_derivative_jump_detected(self):
        a, b = valid_pair()
        b = Span.from_values(b.left, b.right, [["1/4", "2/3", "1/4"]], b.x)
        report = check_boundaries([a, b], Boundaries.from_values([0], [1], 0, 1))
        self.assertTrue(any(c.name == "knot_0_q_d1" and c.status == "VIOLATED" for c in report.checks))

    def test_knot_x_derivative_jump_detected(self):
        a, b = valid_pair()
        b = Span.from_values(b.left, b.right, b.q, ["3/4", "2/3", "-1/4"])
        report = check_boundaries([a, b], Boundaries.from_values([0], [1], 0, 1))
        self.assertTrue(any(c.name == "knot_0_x_d1" and c.status == "VIOLATED" for c in report.checks))

    def test_nonintegrable_quadratic_zero(self):
        p = Span.from_values(0, 1, [[0, 1]], [0, 0, 1])
        report = check_boundaries([p], Boundaries.from_values([0], [1], 0, 1))
        self.assertEqual(report.status, "VIOLATED")
        self.assertTrue(any(c.name == "duration_endpoint_0" and c.status == "VIOLATED" for c in report.checks))

    def test_negative_speed_near_boundary(self):
        p = Span.from_values(0, 1, [[0, 1]], [0, -1, 2])
        report = check_boundaries([p], Boundaries.from_values([0], [1], 0, 1))
        self.assertEqual(report.status, "VIOLATED")

    def test_nonadjacent_spans(self):
        a, b = valid_pair()
        b = Span.from_values("2/3", 1, b.q, b.x)
        result = check_boundaries([a, b], Boundaries.from_values([0], [1], 0, 1))
        self.assertEqual(result.status, "VIOLATED")

    def test_negative_internal_knot_speed_is_violation(self):
        a, b = valid_pair()
        a = Span.from_values(a.left, a.right, a.q, [1, -4])
        b = Span.from_values(b.left, b.right, b.q, [-1, 0])
        result = check_boundaries([a, b], Boundaries.from_values([0], [1], 1, -1))
        self.assertTrue(any(c.name == "knot_0_moving" and c.status == "VIOLATED" for c in result.checks))

    def test_float_exactness(self):
        self.assertEqual(rational(0.1), F.from_float(0.1))
        self.assertNotEqual(rational(0.1), F(1, 10))

    def test_nan_inf_and_degree_guard(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                Span.from_values(0, 1, [[invalid]], [1])
        with self.assertRaises(ValueError):
            Span.from_values(0, 1, [[0, 0, 0, 0, 0]], [1])

    def test_global_derivative_scale(self):
        p = Span.from_values(0, "1/4", [[0, 0, 1]], [1])
        self.assertEqual(p.at("q", 1, 1), (F(8),))
        self.assertEqual(p.at("q", 1, 2), (F(32),))

    def test_q_second_derivative_discontinuity(self):
        a, b = valid_pair()
        b = Span.from_values(b.left, b.right, [["1/4", "1/2", "1/3"]], b.x)
        report = check_boundaries([a, b], Boundaries.from_values([0], [1], 0, 1))
        self.assertTrue(any(c.name == "knot_0_q_d2" and c.status == "VIOLATED" for c in report.checks))

    def test_no_substitute_for_interior_certification(self):
        # Negative x in the interior does not invalidate the limited label PASS:
        # the caller MUST independently check full-domain x positivity.
        p = Span.from_values(0, 1, [[0, 1]], [1, -5, 5])
        self.assertEqual(check_boundaries([p], Boundaries.from_values([0], [1], 1, 1)).status, "PASS")


if __name__ == "__main__":
    unittest.main()
