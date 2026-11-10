import sys, unittest
from fractions import Fraction as F
sys.path.insert(0,'src')
from curobo_certify.bernstein import *
from curobo_certify.interval import Interval

def mid(i): return (i.lo+i.hi)/2
class BernsteinTests(unittest.TestCase):
 def test_bernstein_known_hidden_peak_and_split(self):
  g=[Interval.point(-1),Interval.point(.2),Interval.point(-1)]
  l,r=split(g)
  self.assertTrue(all(c.hi<0 for c in l+r))
 def test_split_identity_rational(self):
  p=[F(1,7),F(-2,3),F(5,4),F(-1,2)]
  b=power_to_bernstein(p)
  l,r=split(b)
  for u in [F(0),F(1,5),F(1,2),F(4,5),F(1)]:
   source=sum(p[i]*u**i for i in range(len(p)))
   child, v=(l,F(2)*u) if u<=F(1,2) else (r,F(2)*u-1)
   val=sum(mid(child[i])*F(comb(len(child)-1,i))*float(v)**i*float(1-v)**(len(child)-1-i) for i in range(len(child)))
   self.assertAlmostEqual(val,float(source),places=12)
 def test_derivative_global_span(self):
  # p(u)=u^3 over global span length 1/8 has dp/ds = 24u^2.
  b=power_to_bernstein([0,0,0,1]); d=derivative(b,F(1,8))
  self.assertAlmostEqual(mid(d[-1]),24,places=12)
 def test_product(self):
  a=power_to_bernstein([1,2]); b=power_to_bernstein([3,-1])
  c=product(a,b); p=bernstein_to_power(c)
  expected=[3,5,-2]
  for actual,want in zip(p,expected): self.assertTrue(actual.contains(want))
 def test_tangent_may_remain_inconclusive(self):
  # (u-1/3)^2 has a zero at a non-dyadic point; finite binary split
  # need not make every Bernstein coefficient strictly non-positive.
  b=power_to_bernstein([F(1,9),F(-2,3),F(1)])
  for _ in range(8): b,_=split(b)
  self.assertTrue(any(c.hi>=0 for c in b))
if __name__=='__main__': unittest.main()
