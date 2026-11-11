import sys,unittest
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from export_native_progress_adapter_operands import eval_model,poly_value

class NativeProgressOperandTests(unittest.TestCase):
    def test_polynomial_derivatives_exact(self):
        c=['1/3','-2/5','7/11','1/13'];u=F(2,7);h=F(3,5)
        self.assertEqual(poly_value(c,u,0,h),sum((F(v)*u**k for k,v in enumerate(c)),F(0)))
        self.assertEqual(poly_value(c,u,1,h),sum((k*F(c[k])*u**(k-1) for k in range(1,4)),F(0))/h)
        self.assertEqual(poly_value(c,u,2,h),sum((k*(k-1)*F(c[k])*u**(k-2) for k in range(2,4)),F(0))/h**2)
    def test_internal_progress_selects_right_span_at_knot(self):
        model={'spans':[{'left':'0','right':'1/2','q':[['0','1','0','1']], 'x':['1','0','0','0']},
                        {'left':'1/2','right':'1','q':[['1/2','2','0','2']], 'x':['1','0','0','0']}]}
        got=eval_model(model,0.5,0)
        self.assertEqual(got['s'],0.5)
        self.assertAlmostEqual(got['q1'],4.0)
        self.assertAlmostEqual(got['q3'],96.0)
if __name__=='__main__':unittest.main()
