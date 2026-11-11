#!/usr/bin/env python3
"""Evaluate exact exported q(s),x(s) model at captured CUDA internal progress.

This separates path/timing evaluation from physical-time inversion: s is copied
from the captured device row. The result remains a diagnostic on the serialized
mathematical model, not a bound on native FP32 execution.
"""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import math


def poly_value(c, u, derivative_order=0, span_width=Fraction(1)):
    result=Fraction(0)
    derived=[]
    for k in range(derivative_order,len(c)):
        factor=1
        for z in range(derivative_order): factor*=k-z
        derived.append(Fraction(c[k])*factor)
    for coefficient in reversed(derived): result=result*u+coefficient
    return result/(span_width**derivative_order)


def eval_model(model, s, joint):
    sf=Fraction.from_float(float(s))
    spans=model['spans']
    selected=None
    for i,row in enumerate(spans):
        left,right=Fraction(row['left']),Fraction(row['right'])
        if left <= sf < right or (i==len(spans)-1 and sf==right):
            selected=(row,left,right);break
    if selected is None:
        raise ValueError(f'progress {s} outside exported spline domain')
    row,left,right=selected;h=right-left;u=(sf-left)/h
    q=row['q'][joint];x=row['x']
    return {'s':float(sf),'q1':float(poly_value(q,u,1,h)),
            'q2':float(poly_value(q,u,2,h)),'q3':float(poly_value(q,u,3,h)),
            'x0':float(poly_value(x,u,0,h)),'x1':float(poly_value(x,u,1,h)),
            'x2':float(poly_value(x,u,2,h))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace-analysis',type=Path,required=True)
    p.add_argument('--device-trace',type=Path,required=True)
    p.add_argument('--models-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--native-out',type=Path,required=True)
    a=p.parse_args(); trace_blob=a.trace_analysis.read_bytes();trace=json.loads(trace_blob)
    device_blob=a.device_trace.read_bytes();device_rows=json.loads(device_blob)['records']
    cache={}; rows=[];native_rows=[]
    for n in trace['records']:
        path=n['path_id'];model_path=a.models_dir/f'{path}.json'
        blob=model_path.read_bytes();model=json.loads(blob)
        source=model.get('provenance',{}).get('source_arrays_sha256')
        if source != n['source_arrays_sha256']:
            raise ValueError(f'{path}: source-array SHA mismatch')
        operands=eval_model(model,n['internal_progress_device'],n['joint'])
        row={'path_id':path,'idx':n['trace_idx'],'joint':n['joint'],**operands,
             'jerk':math.sqrt(operands['x0'])*(operands['q3']*operands['x0']+
               1.5*operands['q2']*operands['x1']+0.5*operands['q1']*operands['x2']),
             'physical_time_s':n['physical_time_s'],'native_progress':n['internal_progress_device'],
             'source_arrays_sha256':source,'model_sha256':hashlib.sha256(blob).hexdigest()}
        rows.append(row)
        candidates=[d for d in device_rows if d['idx']==n['trace_idx'] and d['joint']==n['joint']
                    and abs(float(d['t'])-float(n['physical_time_s']))<1e-8]
        if not candidates: raise ValueError(f'{path} idx={n["trace_idx"]}: captured device row absent')
        candidates.sort(key=lambda d:abs(d['s']-n['internal_progress_device'])+
          sum(abs(d[f'x{k}']-n['x_device'][k]) for k in range(3))+
          sum(abs(d[f'q{k}']-n['q_derivatives_device'][k-1]) for k in range(1,4)))
        d=candidates[0]
        native_rows.append({'path_id':path,'idx':n['trace_idx'],'joint':n['joint'],
          's':d['s'],'speed':d['speed'],'q1':d['q1'],'q2':d['q2'],'q3':d['q3'],
          'x0':d['x0'],'x1':d['x1'],'x2':d['x2'],'native_jerk':n['jerk_native_pristine'],
          'physical_time_s':n['physical_time_s'],'source_arrays_sha256':source,
          'device_trace_sha256':hashlib.sha256(device_blob).hexdigest()})
    output={'scope':'EXACT_SERIALIZED_MODEL_AT_CAPTURED_DEVICE_PROGRESS_DIAGNOSTIC',
            'native_linkage_bound':False,'trace_analysis_sha256':hashlib.sha256(trace_blob).hexdigest(),
            'records':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    native={'scope':'CAPTURED_CUDA_DEVICE_OPERANDS_DIAGNOSTIC','native_output_arrays_bitwise_pristine_match':True,
            'device_trace_sha256':hashlib.sha256(device_blob).hexdigest(),'records':native_rows}
    a.native_out.parent.mkdir(parents=True,exist_ok=True);a.native_out.write_text(json.dumps(native,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'rows':len(rows),'paths':len({r['path_id'] for r in rows}),
                      'out':str(a.out),'scope':output['scope']}))

if __name__=='__main__':main()
