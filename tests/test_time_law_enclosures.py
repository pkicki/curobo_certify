import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
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


class StrictRestRootSubstitution(unittest.TestCase):
    """x(0) = 0 endpoints: simple roots integrable, higher-order roots rejected."""

    def test_simple_root_matches_closed_form(self):
        import math
        from fractions import Fraction as Fr
        from curobo_certify.time_law_enclosures import endpoint_root_integral
        # x(s) = s on [0,1]: T(phi) = 2 sqrt(phi)
        for phi in (Fr(1), Fr(1, 4), Fr(1, 3)):
            lo, hi, st = endpoint_root_integral((Fr(0), Fr(1)), Fr(1), phi)
            self.assertEqual(st, "BOUNDED_EXACT_MODEL")
            exact = 2 * math.sqrt(float(phi))
            self.assertLessEqual(float(lo), exact + 1e-15)
            self.assertGreaterEqual(float(hi), exact - 1e-15)
            self.assertLess(float(hi - lo), 1e-6)
        # x(s) = s (1 + s): T(1) = 2 asinh(1)
        lo, hi, st = endpoint_root_integral((Fr(0), Fr(1), Fr(1)), Fr(1), Fr(1), depth=10)
        self.assertEqual(st, "BOUNDED_EXACT_MODEL")
        self.assertTrue(float(lo) <= 2 * math.asinh(1) <= float(hi))

    def test_higher_order_and_negative_roots(self):
        from fractions import Fraction as Fr
        from curobo_certify.time_law_enclosures import endpoint_root_integral
        self.assertEqual(endpoint_root_integral((Fr(0), Fr(0), Fr(1)), Fr(1), Fr(1))[2], "NONINTEGRABLE_ENDPOINT")
        self.assertEqual(endpoint_root_integral((Fr(0), Fr(-1), Fr(3)), Fr(1), Fr(1))[2], "NEGATIVE_X_AT_ENDPOINT")
        self.assertEqual(endpoint_root_integral((Fr(0), Fr(0)), Fr(1), Fr(1))[2], "IDENTICALLY_ZERO_SPEED")
        with self.assertRaises(ValueError):
            endpoint_root_integral((Fr(1), Fr(1)), Fr(1), Fr(1))


class TimeLawGridTests(unittest.TestCase):
    def test_grid_brackets_contain_high_precision_reference(self):
        import numpy as np
        from fractions import Fraction as Fr
        from curobo_certify import time_law_grid as TG
        from curobo_certify.native_error_bound import exact_timing_table
        rng = np.random.default_rng(5)
        tc = (2 + 30 * rng.random(32)).astype(np.float32)
        g = TG.build_grid(tc, 14)
        self.assertEqual(g.status, "BOUNDED_EXACT_MODEL")
        # Reference: composite 16-point Gauss-Legendre on 512 cells per timing span (binary64)
        tab = exact_timing_table()
        coeffs = [[float(sum(tab[m][3 - p][k] * Fr(float(tc[m + k])) for k in range(4))) for p in range(4)] for m in range(29)]
        gl, gw = np.polynomial.legendre.leggauss(16)
        def T(s):
            total = 0.0
            for m in range(29):
                a, b = m / 29, min((m + 1) / 29, s)
                if b <= a:
                    break
                e = np.linspace(a, b, 513); hh = np.diff(e)
                pts = (e[:-1, None] + hh[:, None] * 0.5 * (gl + 1)) - m / 29
                c = coeffs[m]
                x = ((c[3] * pts + c[2]) * pts + c[1]) * pts + c[0]
                total += float(np.sum(hh[:, None] * 0.5 * gw / np.sqrt(x)))
            return total
        for k in (0, 1, 777, 2**13, 12345, 2**14):
            ref = T(k / 2**14)
            self.assertLessEqual(g.T_lo[k], ref + 1e-12)
            self.assertGreaterEqual(g.T_hi[k], ref - 1e-12)
        for t in (0.0, g.T_lo[-1] * 0.3, g.T_lo[-1] * 0.999):
            lo, hi = TG.inverse_bracket(g, float(t))
            self.assertLessEqual(lo, hi)
            if t > 0:
                self.assertLessEqual(g.T_lo[int(np.floor(lo * 2**14))], t + 1e-12)

    def test_nonpositive_x_fails_closed(self):
        import numpy as np
        from curobo_certify import time_law_grid as TG
        tc = np.full(32, 5.0, dtype=np.float32); tc[10:14] = 0
        self.assertEqual(TG.build_grid(tc, 10).status, "UNKNOWN_POSITIVITY")
