"""Sound rational integral/inverse enclosures: mathematical model ONLY."""
from fractions import Fraction as F
from math import sqrt
import unittest

from curobo_certify.time_law_enclosures import (
    inv_sqrt_bracket, power_to_bernstein, subdivide_at,
    Piece, time_prefix, inverse_time_bracket)


class ExactTimeLawTests(unittest.TestCase):
    def test_integer_square_root_bounds(self):
        for x in (F(1),F(4),F(1,4),F(2),F(1,3),F(10**-8),F(1,10**23)):
            if not isinstance(x,F):
                x=F(x)
            low,high=inv_sqrt_bracket(x)
            self.assertLessEqual(low*low*x,1)
            self.assertGreaterEqual(high*high*x,1)
            self.assertLessEqual(low,high)
        self.assertEqual(inv_sqrt_bracket(F(4)),(F(1,2),F(1,2)))
        self.assertEqual(inv_sqrt_bracket(F(1)),(F(1),F(1)))
        for invalid in (F(0),F(-1)):
            with self.assertRaises(ValueError):
                inv_sqrt_bracket(invalid)

    def test_subdivision_preserves_exact_bernstein_polynomial(self):
        b=power_to_bernstein((F(1),F(2),F(3)))
        left,right=subdivide_at(b,F(1,3))
        self.assertEqual(left[0],b[0])
        self.assertEqual(left[-1],right[0])
        self.assertEqual(right[-1],b[-1])
        self.assertEqual(left[-1],F(1)+F(2,3)+F(1,3))

    def test_constant_speed_exact_integral(self):
        p=(Piece.from_power("0","1",["1"]),)
        for progress in (F(0),F(1,7),F(1,2),F(3,4),F(1)):
            bounds=time_prefix(p,progress)
            self.assertEqual(bounds.status,"BOUNDED_EXACT_MODEL")
            self.assertEqual(bounds.lower,progress)
            self.assertEqual(bounds.upper,progress)
            self.assertEqual(bounds.width,0)

    def test_two_spans_with_different_constant_speeds(self):
        p=(Piece.from_power("0","1/2",["4"]),
           Piece.from_power("1/2","1",["1"]))
        end=time_prefix(p,F(1))
        self.assertEqual(end.lower,F(3,4))
        self.assertEqual(end.upper,F(3,4))
        mid=time_prefix(p,F(3,4))
        self.assertEqual(mid.lower,F(1,2))
        self.assertEqual(mid.upper,F(1,2))
        inv=inverse_time_bracket(p,F(1,2),max_steps=20)
        self.assertEqual(inv.status,"BRACKETED_EXACT_MODEL")
        self.assertEqual(inv.lower_s,inv.upper_s)
        self.assertEqual(inv.lower_s,F(3,4))

    def test_nonconstant_positive_x_time_brackets_are_rigorous(self):
        # x(s)=1+s on [0,1], exact duration =2(sqrt(2)-1).
        p=(Piece.from_power("0","1",["1","1"]),)
        small=time_prefix(p,F(1),depth=2)
        tight=time_prefix(p,F(1),depth=8)
        exact_double=2*(sqrt(2)-1)
        self.assertLessEqual(float(small.lower),exact_double)
        self.assertGreaterEqual(float(small.upper),exact_double)
        self.assertLessEqual(float(tight.lower),exact_double)
        self.assertGreaterEqual(float(tight.upper),exact_double)
        self.assertLessEqual(tight.width,small.width)
        self.assertEqual(tight.scope,"EXACT_SERIALIZED_POSITIVE_X_TIME_LAW_ONLY_NOT_NATIVE_INVERSION")

    def test_inverse_of_constant_speed(self):
        p=(Piece.from_power("0","1",["1"]),)
        inv=inverse_time_bracket(p,F(1,4))
        self.assertEqual(inv.status,"BRACKETED_EXACT_MODEL")
        self.assertEqual(inv.lower_s,F(1,4))
        self.assertEqual(inv.upper_s,F(1,4))

    def test_strict_rest_endpoint_is_not_silently_accepted(self):
        # Positive for interior s but zero at start. Requires a separate
        # analytic endpoint-integrability proof, intentionally unavailable here.
        p=(Piece.from_power("0","1",["0","1"]),)
        got=time_prefix(p,F(1),depth=6)
        self.assertEqual(got.status,"UNKNOWN_POSITIVITY")
        inv=inverse_time_bracket(p,F(1,4))
        self.assertEqual(inv.status,"UNKNOWN_POSITIVITY")

    def test_bad_inputs_and_noncontiguous_segments_fail(self):
        with self.assertRaises(ValueError):
            Piece.from_power("1","0",["1"])
        with self.assertRaises(ValueError):
            power_to_bernstein((F(1),)*5)
        with self.assertRaises(ValueError):
            subdivide_at((F(1),F(2)),F(2))
        p=(Piece.from_power("0","1/2",["1"]),
           Piece.from_power("3/4","1",["1"]))
        with self.assertRaises(ValueError):
            time_prefix(p,F(1))
        with self.assertRaises(ValueError):
            inverse_time_bracket((Piece.from_power("0","1",["1"]),),F(-1))


if __name__=="__main__":
    unittest.main()
