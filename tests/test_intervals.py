import sys, unittest, math
from fractions import Fraction as F
sys.path.insert(0,'src')
from curobo_certify.interval import *
class IntervalTests(unittest.TestCase):
 def test_fraction_enclosure(self):
  i=Interval.rational(F(1,10)); self.assertLessEqual(F.from_float(i.lo),F(1,10)); self.assertGreaterEqual(F.from_float(i.hi),F(1,10))
 def test_exact_dyadic(self): self.assertEqual(Interval.rational(F(3,8)),Interval(0.375,0.375))
 def test_all_exact_integer_products_enclosed(self):
  for a in range(-8,9):
   for b in range(-8,9): self.assertTrue((Interval.point(a)*Interval.point(b)).contains(a*b))
 def test_zero_divisor_rejected(self):
  with self.assertRaises(ZeroDivisionError): Interval(-1,1).reciprocal()
 def test_nan_rejected_and_indeterminate_conservative(self):
  with self.assertRaises(ValueError): Interval(float('nan'),1)
  self.assertEqual(Interval(0,0)*Interval(float('-inf'),float('inf')),Interval(float('-inf'),float('inf')))
if __name__=='__main__': unittest.main()
