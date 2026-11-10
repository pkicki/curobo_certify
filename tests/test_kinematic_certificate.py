import sys, unittest
sys.path.insert(0,'src')
from curobo_certify.certify_cpu import certify_span
from curobo_certify.interval import Interval
from curobo_certify.residuals import k3_residuals
from curobo_certify.bernstein import power_to_bernstein

class K3Tests(unittest.TestCase):
 def test_constant_speed_straight_path_certifies_kinematics_but_not_full(self):
  # q=s, x=1, limits all 2 -> velocity residual -3/4; a=j=0.
  r=certify_span([0,1],[1],(2,2,2))
  self.assertEqual(r.k3_status,'UNKNOWN_BOUNDARY'); self.assertEqual(r.full_status,'UNSUPPORTED')
 def test_hidden_velocity_violation_not_sampled_away(self):
  # q'=1, x(s) represented by a positive bump with interior overspeed.
  r=certify_span([0,1],[1,8,1],(1,100,100),max_depth=8)
  self.assertEqual(r.k3_status,'VIOLATED')
 def test_zero_interior_speed_is_not_certified(self):
  # x has a non-dyadic interior zero; strict positivity obligation blocks.
  x=power_to_bernstein([1/9,-2/3,1])
  r=certify_span([0,1],x,(100,100,100),max_depth=5)
  self.assertIn(r.k3_status,('UNKNOWN_DEPTH','UNKNOWN_NODES'))
 def test_node_budget_is_unknown(self):
  r=certify_span([0,1],[1,8,1],(1,100,100),max_depth=8,max_nodes=1)
  self.assertEqual(r.k3_status,'UNKNOWN_NODES')
 def test_residual_coefficients_enclose_constant_case(self):
  r=k3_residuals([0,1],[1],(2,2,2))
  self.assertTrue(all(c.hi<0 for c in r['velocity']))
  self.assertTrue(all(c.hi<0 for n in ('acceleration_positive','acceleration_negative','jerk') for c in r[n]))
if __name__=='__main__': unittest.main()
