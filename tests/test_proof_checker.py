import sys, unittest
sys.path.insert(0,'src')
from curobo_certify.certify_cpu import certify_span
from curobo_certify.proof_check import verify_span
class ProofCheckerTests(unittest.TestCase):
 def test_replays_residual_leaves_but_boundary_stays_unknown(self):
  q,x,lim=[0,1],[1],(2,2,2); r=certify_span(q,x,lim)
  self.assertEqual(r.k3_status,'UNKNOWN_BOUNDARY')
  self.assertEqual(verify_span(q,x,lim,r),(True,'all leaves replayed'))
 def test_rejects_tampered_leaf(self):
  q,x,lim=[0,1],[1],(2,2,2); r=certify_span(q,x,lim)
  r.leaves[0].status='UNKNOWN_DEPTH'
  ok,reason=verify_span(q,x,lim,r); self.assertFalse(ok); self.assertIn('invalid certified leaf',reason)
 def test_rejects_missing_partition_leaf(self):
  q,x,lim=[0,1],[1],(2,2,2); r=certify_span(q,x,lim)
  r.leaves[0].path='0'
  ok,reason=verify_span(q,x,lim,r); self.assertFalse(ok); self.assertIn('cover',reason)
if __name__=='__main__': unittest.main()
