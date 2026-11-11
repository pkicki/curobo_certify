import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from attribute_native_jerk import attribute,compare,ulp_distance,jerk


def row(**overrides):
    d={'idx':0,'joint':0,'s':0.25,'q1':1.5,'q2':1.0,'q3':0.2,
       'x0':1.,'x1':0.5,'x2':0.25}
    d.update(overrides)
    d['native_jerk']=jerk(d,d)
    return d

class AttributionTests(unittest.TestCase):
    def test_identical_inputs(self):
        n=row();a={**n,'jerk':n['native_jerk']}
        result=attribute(n,a)
        self.assertEqual(result['native_minus_adapter'],0)
        self.assertEqual(result['q_operand_effect'],0)
        self.assertEqual(result['x_operand_effect'],0)

    def test_shapley_decomposition_additive(self):
        n=row(q1=1.25,x1=0.85,s=0.250001)
        a=row();a['jerk']=a.pop('native_jerk')
        r=attribute(n,a)
        total=r['q_operand_effect']+r['x_operand_effect']
        self.assertAlmostEqual(total,jerk(n,n)-jerk(a,a))
        self.assertFalse(r['same_progress'])
        self.assertNotEqual(r['operands']['q1']['ulp32'],0)

    def test_single_q_change_has_zero_x_effect(self):
        n=row(q3=0.3)
        a=row();a['jerk']=a.pop('native_jerk')
        result=attribute(n,a)
        self.assertAlmostEqual(result['x_operand_effect'],0)

    def test_float32_ulps(self):
        self.assertEqual(ulp_distance(1.0,1.0),0)
        self.assertEqual(ulp_distance(1.0,1.0+2**-23),1)
        self.assertEqual(ulp_distance(float('nan'),1.0),None)
        self.assertEqual(ulp_distance(0.,-0.),0)

    def test_compare_rejects_missing_or_duplicate(self):
        n=row();a={**n,'jerk':n['native_jerk']}
        with self.assertRaises(ValueError):
            compare({'records':[n,n]},{'records':[a]})
        with self.assertRaises(ValueError):
            compare({'records':[n]},{'records':[]})
        good=compare({'records':[n]},{'records':[a]})
        self.assertEqual(len(good['records']),1)
        self.assertFalse(good['continuous_native_linked_certificate'])

if __name__=='__main__':unittest.main()
