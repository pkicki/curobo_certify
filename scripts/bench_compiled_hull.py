#!/usr/bin/env python3
"""Stage profile of an incomplete compiled hull-only K3 screening component.

No native parity, endpoint/continuity check, independent proof replay or
complete-request TimeOpt work occurs. Never label as a K3 certificate.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from curobo_certify.compiled_hull import CompiledHull, pack_model, partial_summary  # noqa:E402


def profile(blob, compiled, repeats, warmups):
    if repeats < 1 or warmups < 0:
        raise ValueError('bad number of repetitions')
    raw=[]
    for rep in range(warmups+repeats):
        t0=time.perf_counter_ns()
        model=json.loads(blob)
        t1=time.perf_counter_ns()
        packed=pack_model(model)
        t2=time.perf_counter_ns()
        bounds=compiled.evaluate_packed(*packed)
        t3=time.perf_counter_ns()
        summary=partial_summary(model,bounds)
        t4=time.perf_counter_ns()
        if rep >= warmups:
            raw.append({'repeat':rep-warmups,'parse_ns':t1-t0,'pack_ns':t2-t1,
                        'compiled_hull_ns':t3-t2,'summary_ns':t4-t3,
                        'total_ns':t4-t0,'status':summary['status']})
    fields=('parse_ns','pack_ns','compiled_hull_ns','summary_ns','total_ns')
    return {'scope':'EXPLORATORY_SERIALIZED_MODEL_HULL_ONLY',
            'campaign_eligible':False,'native_linked':False,
            'full_k3_certificate':False,'independent_proof_replay':False,
            'source_sha256':hashlib.sha256(blob).hexdigest(),
            'warmups':warmups,'repeats':repeats,
            'p50_ns':{name:statistics.median(r[name] for r in raw) for name in fields},
            'raw':raw}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('models',type=Path,nargs='+')
    p.add_argument('--lib',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warmups',type=int,default=50)
    p.add_argument('--repeats',type=int,default=10)
    args=p.parse_args()
    lib=CompiledHull(args.lib)
    results=[]
    for path in args.models:
        results.append({'model':str(path),**profile(path.read_bytes(),lib,args.repeats,args.warmups)})
    output={'schema_version':1,'experiment':'COMPILED_HULL_STAGE_TIMING_NOT_FULL_CERTIFICATION',
            'excludes':['native solve','GPU','native linkage','boundary and knot checks',
                        'strict positivity','proof replay','complete request'],
            'results':results}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(f'Stage-timed {len(results)} model files -> {args.out}')


if __name__=='__main__':
    main()
