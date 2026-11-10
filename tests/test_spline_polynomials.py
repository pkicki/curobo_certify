import sys
import unittest
from fractions import Fraction

import numpy as np

sys.path.insert(0, "src")

from curobo_certify.boundary_reference import evaluate
from curobo_certify.spline_polynomials import extract_timeopt_spans


class SplinePolynomialExtractionTests(unittest.TestCase):
    def test_linear_cardinal_path_and_constant_timing_on_union_knots(self):
        points = np.arange(5, dtype=np.float32)[:, None]
        timing = np.ones(32, dtype=np.float32)
        spans = extract_timeopt_spans(points, timing)
        self.assertEqual(len(spans), 30)  # union of two geometry spans and 29 timing spans
        for span in spans:
            self.assertEqual(span.q[0][1:], (2 * (span.right - span.left),))
            self.assertEqual(evaluate(span.q[0], Fraction(0)), 1 + 2 * span.left)
            self.assertEqual(evaluate(span.q[0], Fraction(1)), 1 + 2 * span.right)
            self.assertEqual(span.x, (Fraction(1),))

    def test_float32_inputs_are_preserved_as_exact_binary_rationals(self):
        value = np.float32(0.1)
        points = np.full((4, 1), value, dtype=np.float32)
        timing = np.full(32, value, dtype=np.float32)
        spans = extract_timeopt_spans(points, timing)
        exact_value = Fraction.from_float(float(value))
        self.assertTrue(all(span.q == ((exact_value,),) for span in spans))
        self.assertTrue(all(span.x == (exact_value,) for span in spans))

    def test_rejects_wrong_timing_control_count(self):
        with self.assertRaises(ValueError):
            extract_timeopt_spans(np.ones((4, 1), dtype=np.float32), np.ones(31, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
