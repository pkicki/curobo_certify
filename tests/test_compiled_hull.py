"""Differential exact-rational vs compiled conservative binary64 hull tests.

These checks validate individual K3 hulls, not full/native certificates.
"""
import math
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from curobo_certify.compiled_hull import CompiledHull, FAMILIES, rational_enclosure, pack_model, partial_summary
from curobo_certify.boundary_reference import Span
from curobo_certify.exact_k3_reference import build_residuals, bernstein_coeffs
from build_compiled_hull import build
from bench_compiled_hull import profile


@unittest.skipUnless(shutil.which('g++'),'requires C++ compiler')
class CompiledHullTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.temp.name)/'libk3_hull.so'
        cls.manifest=build(cls.path)
        cls.compiled=CompiledHull(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_original_analytic_fixture_exact_enclosure(self):
        import json
        model=json.loads((ROOT/'fixtures/boundary_multispan_analytic.json').read_text())
        self.check_model(model)
        result=partial_summary(model,self.compiled.evaluate(model))
        self.assertEqual(result['status'],'INITIAL_HULLS_NONPOSITIVE_PARTIAL')
        self.assertEqual(result['family']['positive_x']['count'],2)
        self.assertEqual(result['family']['jerk']['count'],2)
        self.assertFalse(result['full_k3_certificate'])

    def check_model(self,model):
        results=self.compiled.evaluate(model)
        joints=len(model['spans'][0]['q'])
        for s,item in enumerate(model['spans']):
            span=Span.from_values(item['left'],item['right'],item['q'],item['x'])
            residuals=list(build_residuals(span,model['limits']['velocity'],
                                          model['limits']['acceleration'],model['limits']['jerk']))
            for fam,joint,power in residuals:
                idx=(s*joints+(0 if joint<0 else joint))*5+FAMILIES.index(fam)
                lower,upper=results[idx]
                target_degree={"positive_x":3,"velocity":7,"acceleration_plus":4,
                               "acceleration_minus":4,"jerk":9}[fam]
                padded=tuple(power)+(F(0),)*(target_degree+1-len(power))
                exact=bernstein_coeffs(padded)
                self.assertTrue(F.from_float(lower)<=min(exact), (fam,s,joint,lower,min(exact)))
                self.assertTrue(F.from_float(upper)>=max(exact), (fam,s,joint,upper,max(exact)))

    def test_256_seeded_multijoint_cases(self):
        rng=random.Random(20261111)
        for _ in range(256):
            joints=2
            span_count=rng.randrange(1,4)
            spans=[]
            left=F(0)
            for i in range(span_count):
                h=F(rng.randrange(1,10),rng.choice((8,16,32)))
                right=left+h
                q=[[str(F(rng.randrange(-4,5),rng.choice((8,16)))) for _ in range(4)]
                   for _ in range(joints)]
                x=[str(F(rng.randrange(-2,6),16)) for _ in range(4)]
                spans.append({'left':str(left),'right':str(right),'q':q,'x':x})
                left=right
            model={'schema_version':1,'basis':'local_power_ascending_u',
                   'spans':spans,'limits':{'velocity':['2','3'],
                                          'acceleration':['5','10'],
                                          'jerk':['20','50']}}
            self.check_model(model)

    def test_zero_and_non_dyadic_rational_input(self):
        for exact in (F(0), F(1,3), F(-1,7), F(10**20,3),F(1,10**310), F(-1,10**310)):
            lo,hi=rational_enclosure(str(exact))
            self.assertLessEqual(F.from_float(lo),exact)
            self.assertGreaterEqual(F.from_float(hi),exact)
        with self.assertRaises(ValueError):
            rational_enclosure(str(10**400))

    def test_inputs_fail_closed(self):
        template={'schema_version':1,'basis':'local_power_ascending_u',
                  'spans':[{'left':0,'right':1,'q':[[0,1]],'x':[1]}],
                  'limits':{'velocity':[2],'acceleration':[2],'jerk':[2]}}
        self.assertEqual(len(self.compiled.evaluate(template)),5)
        for adjust in (
            {'basis':'not-a-polynomial'},
            {'spans':[{'left':0,'right':0,'q':[[0,1]],'x':[1]}]},
            {'spans':[{'left':0,'right':1,'q':[[0,1,2,3,4]],'x':[1]}]},
            {'limits':{'velocity':[0],'acceleration':[2],'jerk':[2]}},
        ):
            variant=template|adjust
            with self.assertRaises(ValueError):
                self.compiled.evaluate(variant)

    def test_native_stage_profiling_truthfully_scoped(self):
        raw=(ROOT/'fixtures/boundary_multispan_analytic.json').read_bytes()
        report=profile(raw,self.compiled,repeats=4,warmups=1)
        self.assertFalse(report['campaign_eligible'])
        self.assertFalse(report['full_k3_certificate'])
        self.assertEqual(len(report['raw']),4)
        self.assertTrue(all(r['total_ns']>0 for r in report['raw']))
        self.assertEqual(report['raw'][0]['total_ns'],sum(report['raw'][0][k] for k in
                         ('parse_ns','pack_ns','compiled_hull_ns','summary_ns')))


if __name__=='__main__':
    unittest.main()
