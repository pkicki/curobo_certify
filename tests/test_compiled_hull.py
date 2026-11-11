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

    def test_10000_seeded_individual_coefficient_enclosures(self):
        """Differential oracle checks every coefficient, not only hull extrema."""
        rng=random.Random(0x6B334B45524E)
        spans=[]
        left=F(0)
        for i in range(10000):
            # Mix dyadic/non-dyadic values, cancellation, tiny but representable
            # spans, zero endpoints, nearly active limits and ordinary cases.
            den=rng.choice((3,5,7,10,16,31,64,127))
            h=F(rng.randrange(1,20),den)
            if i%17==0: h=F(1,10**rng.randrange(8,22))
            q=[]
            for k in range(4):
                numer=rng.randrange(-1000,1001)
                q.append(str(F(numer,rng.choice((3,5,16,127)))))
            x=[str(F(rng.randrange(-50,101),rng.choice((3,7,32,127)))) for _ in range(4)]
            if i%11==0: x[0]='0'
            if i%13==0: x=['0','1','-1','0']  # interior speed zero
            spans.append({'left':str(left),'right':str(left+h),
                          'q':[q],'x':x})
            left+=h
        model={'schema_version':1,'basis':'local_power_ascending_u','spans':spans,
               'limits':{'velocity':['3.7000000001'],
                         'acceleration':['6.3000000001'],'jerk':['40.1000000001']}}
        coeffs=self.compiled.evaluate_coefficients(model)
        degrees={'positive_x':3,'velocity':7,'acceleration_plus':4,
                 'acceleration_minus':4,'jerk':9}
        offsets={'positive_x':0,'velocity':4,'acceleration_plus':12,
                 'acceleration_minus':17,'jerk':22}
        for s,item in enumerate(spans):
            span=Span.from_values(item['left'],item['right'],item['q'],item['x'])
            for family,joint,power in build_residuals(span,['3.7000000001'],
                    ['6.3000000001'],['40.1000000001']):
                padded=tuple(power)+(F(0),)*(degrees[family]+1-len(power))
                exact=bernstein_coeffs(padded)
                base=s*32+offsets[family]
                for k,target in enumerate(exact):
                    lo,hi=coeffs[base+k]
                    self.assertLessEqual(F.from_float(lo),target,
                                         (s,family,k,'lower',lo,target))
                    self.assertGreaterEqual(F.from_float(hi),target,
                                            (s,family,k,'upper',hi,target))

    def test_zero_and_non_dyadic_rational_input(self):
        for exact in (F(0), F(1,3), F(-1,7), F(10**20,3),F(1,10**310), F(-1,10**310)):
            lo,hi=rational_enclosure(str(exact))
            self.assertLessEqual(F.from_float(lo),exact)
            self.assertGreaterEqual(F.from_float(hi),exact)
        with self.assertRaises(ValueError):
            rational_enclosure(str(10**400))

    def test_near_active_signed_families_and_speed_zeros_are_partial(self):
        models=[]
        for q,limits,x in (
            ([['0','1']],{'velocity':['1'],'acceleration':['2'],'jerk':['2']},['1']),
            ([['0','0','0','1/6']],{'velocity':['2'],'acceleration':['1'],'jerk':['1']},['1']),
            ([['0','0','-1/2']],{'velocity':['2'],'acceleration':['1'],'jerk':['2']},['1']),
        ):
            models.append({'schema_version':1,'basis':'local_power_ascending_u',
               'spans':[{'left':'0','right':'1','q':q,'x':x}],
               'limits':limits})
        for model in models: self.check_model(model)
        # Endpoint zeros may remain inconclusive due outward cancellation; in
        # all cases this component is forbidden from accepting traversal.
        endpoint={'schema_version':1,'basis':'local_power_ascending_u',
          'spans':[{'left':'0','right':'1','q':[['0']],'x':['0','1','-1']}],
          'limits':{'velocity':['2'],'acceleration':['2'],'jerk':['2']}}
        out=partial_summary(endpoint,self.compiled.evaluate(endpoint))
        self.assertEqual(out['status'],'INITIAL_HULL_UNKNOWN_OR_VIOLATED')
        self.assertFalse(out['full_k3_certificate'])
        interior=endpoint|{'spans':[{'left':'0','right':'1','q':[['0']],
                                     'x':['1/4','-1','1']}]}
        self.assertEqual(partial_summary(interior,self.compiled.evaluate(interior))['status'],
                         'INITIAL_HULL_UNKNOWN_OR_VIOLATED')

    def test_overflow_and_nonfinite_inputs_fail_closed(self):
        base={'schema_version':1,'basis':'local_power_ascending_u',
              'spans':[{'left':'0','right':'1','q':[['0','0','0','1']],
                        'x':['1']}],
              'limits':{'velocity':['2'],'acceleration':['2'],'jerk':['2']}}
        huge_span=base|{'spans':[{'left':'0','right':'1/10**100','q':[['0','0','0','1']],
                                  'x':['1']}]}
        # The literal schema does not evaluate expressions.
        huge_span['spans'][0]['right']=str(F(1,10**100))
        with self.assertRaises(ValueError): self.compiled.evaluate(huge_span)
        for value in (float('nan'),float('inf'),-float('inf')):
            bad=base|{'spans':[{'left':0,'right':1,'q':[[0,1]],'x':[value]}]}
            with self.assertRaises((ValueError,OverflowError,ZeroDivisionError)):
                self.compiled.evaluate(bad)

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

    def test_non_nearest_runtime_rounding_fails_closed(self):
        import ctypes
        model={'schema_version':1,'basis':'local_power_ascending_u',
               'spans':[{'left':0,'right':1,'q':[[0,1]],'x':[1]}],
               'limits':{'velocity':[2],'acceleration':[2],'jerk':[2]}}
        packed=self.compiled.evaluate(model)  # also validates nearest baseline
        del packed
        n,d,arrays=pack_model(model)
        libc=ctypes.CDLL(None)
        libc.fegetround.restype=ctypes.c_int
        libc.fesetround.argtypes=[ctypes.c_int]
        old=libc.fegetround()
        try:
            for mode in (0x400,0x800,0xc00):  # glibc x86 FE_DOWN/UP/TOWARDZERO
                self.assertEqual(libc.fesetround(mode),0)
                with self.assertRaisesRegex(ValueError,'status -2'):
                    self.compiled.evaluate_packed(n,d,arrays)
        finally:
            libc.fesetround(old)

    def test_flush_to_zero_and_denormals_are_rejected(self):
        import ctypes
        source='#include <xmmintrin.h>\nunsigned getcsr(){return _mm_getcsr();}\nvoid setcsr(unsigned x){_mm_setcsr(x);}\n'
        helper=Path(self.temp.name)/'mxcsr.c';shared=Path(self.temp.name)/'mxcsr.so'
        helper.write_text(source)
        subprocess.run(['gcc','-shared','-fPIC',str(helper),'-o',str(shared)],check=True)
        mx=ctypes.CDLL(str(shared));mx.getcsr.restype=ctypes.c_uint;mx.setcsr.argtypes=[ctypes.c_uint]
        model={'schema_version':1,'basis':'local_power_ascending_u',
               'spans':[{'left':0,'right':1,'q':[[0,1]],'x':[1]}],
               'limits':{'velocity':[2],'acceleration':[2],'jerk':[2]}}
        n,d,arrays=pack_model(model);old=mx.getcsr()
        try:
            for bit in (1<<15,1<<6):
                mx.setcsr(old|bit)
                with self.assertRaisesRegex(ValueError,'status -2'):
                    self.compiled.evaluate_packed(n,d,arrays)
        finally:
            mx.setcsr(old)

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
