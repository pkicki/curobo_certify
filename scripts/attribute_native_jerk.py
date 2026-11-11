#!/usr/bin/env python3
"""Attribute observed native-vs-adapter jerk differences from captured operands.

This is an empirical diagnostic, NOT a continuous-time error bound. Expected
inputs: native trace JSON from parse_timeopt_cuda_trace.py (records include
idx/joint, q1/q2/q3, x0/x1/x2, s, native_jerk), and adapter JSON records
with the same idx/joint and fields s,q1,q2,q3,x0,x1,x2,jerk. Both records
must correspond to the same requested physical times and original solutions.

q contribution and x contribution are symmetric Shapley decompositions of
the mathematical jerk equation, not floating-point operational attributions.
Residual against native includes native FP32 arithmetic/ordering effects.
"""
import argparse
import json
import math
import struct
from pathlib import Path

OPERANDS=('q1','q2','q3','x0','x1','x2')


def bits(x):
    return int.from_bytes(struct.pack('!f',x),'big')


def ulp_distance(a,b):
    """IEEE binary32 ordered ULP distance, after rounding both values to FP32."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    try:
        aa,bb=bits(a),bits(b)
    except OverflowError:
        return None
    if a == b:  # IEEE +0 and -0 represent the same real number
        return 0
    def ordered(x):
        return ((~x) & 0xffffffff) if x & 0x80000000 else (x | 0x80000000)
    return abs(ordered(aa)-ordered(bb))


def jerk(q,x):
    if x['x0'] < 0:
        raise ValueError('negative squared speed')
    return math.sqrt(x['x0'])*(q['q3']*x['x0']+1.5*q['q2']*x['x1']+0.5*q['q1']*x['x2'])


def attribute(native,adapter):
    for key in (*OPERANDS,'s'):
        if not isinstance(native.get(key), (float,int)) or not isinstance(adapter.get(key),(float,int)):
            raise ValueError('missing numerical operand: '+key)
        if not (math.isfinite(native[key]) and math.isfinite(adapter[key])):
            raise ValueError('nonfinite operand: '+key)
    n, a = native,adapter
    f_aa=jerk(a,a)
    f_nn=jerk(n,n)
    f_na=jerk(n,a)
    f_an=jerk(a,n)
    q_effect=(f_na-f_aa+f_nn-f_an)/2
    x_effect=(f_an-f_aa+f_nn-f_na)/2
    if not math.isclose(q_effect+x_effect, f_nn-f_aa, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError('decomposition mismatch')
    return {'idx':n['idx'],'joint':n['joint'],'source':'OBSERVED_OPERAND_DIAGNOSTIC',
            'native_jerk':n['native_jerk'],'adapter_jerk':a['jerk'],
            'native_minus_adapter':n['native_jerk']-a['jerk'],
            'q_operand_effect':q_effect,'x_operand_effect':x_effect,
            'native_fp32_final_arithmetic_residual':n['native_jerk']-f_nn,
            'adapter_final_arithmetic_residual':a['jerk']-f_aa,
            'progress_difference':n['s']-a['s'],
            'same_progress':n['s']==a['s'],
            'operands':{k:{'native':n[k],'adapter':a[k],
                            'difference':n[k]-a[k],'ulp32':ulp_distance(n[k],a[k])}
                        for k in OPERANDS}}


def compare(native_data,adapter_data):
    if not isinstance(native_data,dict) or not isinstance(adapter_data,dict):
        raise ValueError('expected native and adapter JSON objects')
    n={}
    for row in native_data['records']:
        key=(row['idx'],row['joint'])
        if key in n:raise ValueError('duplicate native record')
        n[key]=row
    a={}
    for row in adapter_data['records']:
        key=(row['idx'],row['joint'])
        if key in a:raise ValueError('duplicate adapter record')
        a[key]=row
    if n.keys()!=a.keys() or not n:
        raise ValueError('native and adapter capture keys must match exactly and be nonempty')
    return {'scope':'EMPIRICAL_NATIVE_OPERAND_DECOMPOSITION_NOT_ERROR_BOUND',
            'continuous_native_linked_certificate':False,
            'records':[attribute(n[k],a[k]) for k in sorted(n)]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('native_trace',type=Path)
    p.add_argument('adapter_operands',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    data=compare(json.loads(args.native_trace.read_text()),
                 json.loads(args.adapter_operands.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(f'Compared {len(data["records"])} native-operand records -> {args.out}')


if __name__=='__main__':
    main()
