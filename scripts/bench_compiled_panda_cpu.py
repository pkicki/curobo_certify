#!/usr/bin/env python3
"""Paired exploratory four-arm CPU profile on frozen real Panda NPZ outputs.

All arms are mathematical serialized-model experiments, not native-linked or
complete TimeOpt request latency. Source NPZ parse/extraction is inside timers.
"""
from __future__ import annotations
import argparse,hashlib,json,os,platform,random,statistics,sys,time
from fractions import Fraction
from io import BytesIO
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from curobo_certify.boundary_reference import Boundaries
from curobo_certify.spline_polynomials import extract_timeopt_spans
from curobo_certify.bernstein import power_to_bernstein
from curobo_certify.residuals import k3_residuals
from curobo_certify.piecewise_cpu import certify_piecewise_k3,replay_piecewise_k3
from curobo_certify.compiled_hull import CompiledHull,pack_model,partial_summary

CASES=('python_full','python_hull_only','compiled_hull_only','compiled_plus_full_replay')

def exact_model(arrays,spans,path_id):
    limits={name:np.minimum(np.abs(arrays[key][0]),np.abs(arrays[key][1])).astype(float).tolist()
            for name,key in (('velocity','velocity_limits'),('acceleration','acceleration_limits'),('jerk','jerk_limits'))}
    rows=[{'left':str(s.left),'right':str(s.right),
           'q':[[str(c) for c in poly] for poly in s.q],
           'x':[str(c) for c in s.x]} for s in spans]
    return {'schema_version':1,'basis':'local_power_ascending_u','spans':rows,'limits':limits,
            'provenance':{'path_id':path_id,'source_arrays_sha256':hashlib.sha256(arrays['_raw']).hexdigest()}}

def load_extract(blob,path_id):
    a=np.load(BytesIO(blob),allow_pickle=False)
    arrays={k:a[k].copy() for k in a.files};a.close();arrays['_raw']=blob
    spans=extract_timeopt_spans(arrays['effective_geometry'],arrays['timing_controls'])
    limits=[np.minimum(np.abs(arrays[k][0]),np.abs(arrays[k][1])).astype(float).tolist()
             for k in ('velocity_limits','acceleration_limits','jerk_limits')]
    targets=Boundaries.from_values(q_start=arrays['requested_start'][0].astype(float).tolist(),
       q_end=arrays['requested_goal'][0].astype(float).tolist(),x_start=spans[0].at('x',0)[0],
       x_end=spans[-1].at('x',1)[0],strict_rest=True)
    return arrays,spans,limits,targets

def python_hull(spans,limits):
    counts={'positive_x':0,'velocity':0,'acceleration_positive':0,'acceleration_negative':0,'jerk':0}
    passes={k:0 for k in counts}; upper={k:-float('inf') for k in counts}
    residual_ns=0;check_ns=0
    for span in spans:
        t=time.perf_counter_ns()
        x=power_to_bernstein(span.x,degree=3)
        xpos=[-v for v in x]
        upper['positive_x']=max(upper['positive_x'],max(v.hi for v in xpos));counts['positive_x']+=1
        passes['positive_x']+=int(all(v.hi<=0 for v in xpos))
        residual_ns += time.perf_counter_ns()-t
        for joint,q in enumerate(span.q):
            t=time.perf_counter_ns()
            qb=power_to_bernstein(q,degree=3)
            res=k3_residuals(qb,x,[limits[0][joint],limits[1][joint],limits[2][joint]],span.right-span.left)
            residual_ns += time.perf_counter_ns()-t
            t=time.perf_counter_ns()
            mapping={'velocity':'velocity','acceleration_positive':'acceleration_positive',
                     'acceleration_negative':'acceleration_negative','jerk':'jerk'}
            for family,coeff in res.items():
                counts[family]+=1;upper[family]=max(upper[family],max(v.hi for v in coeff))
                passes[family]+=int(all(v.hi<=0 for v in coeff))
            check_ns += time.perf_counter_ns()-t
    return {'counts':counts,'hull_passes':passes,'max_upper':upper,
            'partial_only':True,'status':'INITIAL_HULL_PARTIAL',
            'residual_and_bernstein_ns':residual_ns,'hull_check_ns':check_ns}

def full_proof(spans,limits,targets):
    t=time.perf_counter_ns()
    result=certify_piecewise_k3(spans,targets,*limits,max_depth=12,max_nodes=8192)
    certify_ns=time.perf_counter_ns()-t
    t=time.perf_counter_ns()
    ok,detail=replay_piecewise_k3(spans,targets,*limits,result)
    replay_ns=time.perf_counter_ns()-t
    if result.k3_status=='CERTIFIED_K3' and not ok: raise RuntimeError('proof replay rejected output: '+detail)
    return {'status':result.k3_status if ok or result.k3_status!='CERTIFIED_K3' else 'REPLAY_FAILED',
            'full_contract_status':result.full_contract_status,'boundary_status':result.boundary_status,
            'nodes':result.processed_nodes,'leaves':sum(len(x.leaves) for row in result.span_results for x in row),
            'proof_replay_pass':ok,'detail':detail,'certify_ns':certify_ns,'replay_ns':replay_ns}

def timed_case(case,source,path_id,compiled):
    stage={};t=time.perf_counter_ns();blob=source.read_bytes() if isinstance(source,Path) else bytes(source);stage['source_read_ns']=time.perf_counter_ns()-t
    t=time.perf_counter_ns();arrays,spans,limits,targets=load_extract(blob,path_id);stage['npz_parse_and_exact_spline_extraction_ns']=time.perf_counter_ns()-t
    model=None;packed=None;compiled_result=None;hull=None;proof=None
    if case in ('python_full','compiled_plus_full_replay'):
        if case=='compiled_plus_full_replay':
            t=time.perf_counter_ns();model=exact_model(arrays,spans,path_id);stage['exact_model_coefficient_construction_ns']=time.perf_counter_ns()-t
            t=time.perf_counter_ns();packed=pack_model(model);stage['rational_interval_packing_ns']=time.perf_counter_ns()-t
            t=time.perf_counter_ns();bounds=compiled.evaluate_packed(*packed);stage['compiled_residual_and_bernstein_hull_ns']=time.perf_counter_ns()-t
            t=time.perf_counter_ns();hull=partial_summary(model,bounds);stage['compiled_hull_summary_ns']=time.perf_counter_ns()-t
        proof=full_proof(spans,limits,targets)
        stage['boundary_residual_and_full_proof_construction_ns']=proof['certify_ns']
        stage['proof_replay_ns']=proof['replay_ns']
        status=proof['status']
    elif case=='python_hull_only':
        t=time.perf_counter_ns();hull=python_hull(spans,limits);stage['python_residual_and_bernstein_ns']=hull.pop('residual_and_bernstein_ns');stage['python_hull_check_ns']=hull.pop('hull_check_ns')
        status=hull['status']
    elif case=='compiled_hull_only':
        t=time.perf_counter_ns();model=exact_model(arrays,spans,path_id);stage['exact_model_coefficient_construction_ns']=time.perf_counter_ns()-t
        t=time.perf_counter_ns();packed=pack_model(model);stage['rational_interval_packing_ns']=time.perf_counter_ns()-t
        t=time.perf_counter_ns();bounds=compiled.evaluate_packed(*packed);stage['compiled_residual_and_bernstein_hull_ns']=time.perf_counter_ns()-t
        t=time.perf_counter_ns();hull=partial_summary(model,bounds);stage['compiled_hull_summary_ns']=time.perf_counter_ns()-t
        status=hull['status']
    else:raise ValueError(case)
    t=time.perf_counter_ns();encoded=json.dumps({'case':case,'status':status,'hull':hull,'proof':proof},sort_keys=True);stage['result_serialization_ns']=time.perf_counter_ns()-t
    stage['total_ns']=sum(stage.values())
    return {'case':case,'span_count':len(spans),'status':status,'hull_status':hull['status'] if hull else None,
            'proof_status':proof['status'] if proof else None,'proof_replay_pass':proof['proof_replay_pass'] if proof else None,
            'hull_detail':hull,'proof_detail':proof,'stage_ns':stage}

def quantile(values,p):
    z=sorted(values);x=p*(len(z)-1);i=int(x);return z[i]+(z[min(i+1,len(z)-1)]-z[i])*(x-i)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-dir',type=Path,default=Path('runs/native_pilot_resolved'))
    ap.add_argument('--lib',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--build-manifest',type=Path)
    ap.add_argument('--warmups',type=int,default=50);ap.add_argument('--repeats',type=int,default=10);ap.add_argument('--seed',type=int,default=20261111)
    a=ap.parse_args();compiled=CompiledHull(a.lib);rng=random.Random(a.seed);paths=[]
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    output={'schema_version':1,'experiment':'FOUR_ARM_PAIRED_COMPILED_HULL_PROFILE','scope':'EXPLORATORY_FROZEN_MODEL_CPU_ONLY',
      'campaign_eligible':False,'native_linked':False,'full_k3_claim_from_compiled':False,
      'case_definitions':{'python_full':'outward-rounded K3 construction, boundaries, proof and replay',
       'python_hull_only':'existing outward Python residual/Bernstein initial hull only',
       'compiled_hull_only':'exact coefficient packing plus C++ initial hull only',
       'compiled_plus_full_replay':'compiled initial hull plus complete outward K3 proof and independent replay'},
      'environment':{'platform':platform.platform(),'python':platform.python_version(),'cpu_model':platform.processor(),
       'cpu_count':os.cpu_count(),'affinity':affinity,'affinity_count':len(affinity) if affinity is not None else None,
       'compiler_manifest':json.loads(a.build_manifest.read_text()) if a.build_manifest else None},
      'warmups':a.warmups,'repetitions':a.repeats,'seed':a.seed,'paths':paths}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for meta_path in sorted(a.pilot_dir.glob('pilot-panda-*.json')):
        meta=json.loads(meta_path.read_text())
        if meta.get('planner_success') is not True: continue
        npz=(ROOT/meta['arrays_path']).resolve();blob=npz.read_bytes();sha=hashlib.sha256(blob).hexdigest()
        if sha!=meta.get('arrays_sha256'):raise ValueError(f'input SHA mismatch: {npz}')
        cases={case:[] for case in CASES};warm_evidence={}
        for case in CASES:
            for _ in range(a.warmups):
                result=timed_case(case,npz,meta['source_id'],compiled)
                warm_evidence[case]=result
        for rep in range(a.repeats):
            order=list(CASES);rng.shuffle(order)
            for case in order:
                result=timed_case(case,npz,meta['source_id'],compiled)
                cases[case].append({'rep':rep,**result})
        summaries={}
        for case,rows in cases.items():
            totals=[r['stage_ns']['total_ns'] for r in rows]
            stages=sorted({k for r in rows for k in r['stage_ns']})
            summaries[case]={'n':len(rows),'p50_ns':quantile(totals,.5),'p95_ns':quantile(totals,.95),'p99_ns':quantile(totals,.99),
              'stage_p50_ns':{s:quantile([r['stage_ns'].get(s,0) for r in rows],.5) for s in stages},
              'stage_p95_ns':{s:quantile([r['stage_ns'].get(s,0) for r in rows],.95) for s in stages},
              'stage_p99_ns':{s:quantile([r['stage_ns'].get(s,0) for r in rows],.99) for s in stages},
              'statuses':{x:sum(r['status']==x for r in rows) for x in sorted({r['status'] for r in rows})}}
        paths.append({'path_id':meta['source_id'],'source_npz':str(npz),'source_arrays_sha256':sha,
          'source_query_sha256':meta.get('source_query_sha256'),'solver_configuration':meta.get('solver_config_hash',meta.get('solver_configuration')),
          'model_json_sha256':hashlib.sha256((ROOT/'runs/pilot_exact_models'/f"{meta['source_id']}.json").read_bytes()).hexdigest(),
          'span_count':cases[CASES[0]][0]['span_count'],'joint_count':7,'warmups_per_case':a.warmups,'timed_repetitions_per_case':a.repeats,
          'case_summary':summaries,'warmup_status':{k:v['status'] for k,v in warm_evidence.items()},'raw':sum(cases.values(),[]),
          'scope':'EXPLORATORY_FROZEN_MODEL_CPU_ONLY; serialized-model; no native-linked or complete-request latency'})
        temp=a.out.with_suffix(a.out.suffix+'.tmp');temp.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n');temp.replace(a.out)
        print(meta['source_id'],{k:round(v['p50_ns']/1e6,3) for k,v in summaries.items()},flush=True)
    print(f'Wrote {len(paths)} paths to {a.out}')
if __name__=='__main__':main()
