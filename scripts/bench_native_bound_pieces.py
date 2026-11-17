#!/usr/bin/env python3
"""Matched frozen-path sweep of the compiled native arithmetic-bound cost.

Each result uses the original physical actuator limits, untouched native
output and a FULL producer + independent verifier. Different native bound
piece budgets are separately valid conservative models, but the experiment
must not infer correctness from smaller observed errors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from curobo_certify import compiled_k3 as K  # noqa: E402


def physical_limits(a):
    return {name:np.minimum(np.abs(a[key][0]),np.abs(a[key][1])).astype(np.float64)
            for name,key in (("velocity","velocity_limits"),
                             ("acceleration","acceleration_limits"),
                             ("jerk","jerk_limits"))}


def one(producer,verifier,arrays,pieces,div_ulps,sqrt_ulps,threads=1):
    geometry=arrays["effective_geometry"]
    timing=arrays["timing_controls"]
    inputs=(geometry,timing,arrays["requested_start"][0],
            arrays["requested_goal"][0],physical_limits(arrays))
    t0=time.perf_counter_ns()
    tightening=K.native_bounds(producer,geometry,timing,pieces=pieces,
                               div_ulps=div_ulps,sqrt_ulps=sqrt_ulps,threads=threads)
    t1=time.perf_counter_ns()
    if not np.isfinite(tightening).all() or (tightening<0).any():
        raise RuntimeError("invalid native error enclosure")
    result=K.certify_and_verify(producer,verifier,*inputs,tighten=tightening,
                               max_depth=12,max_nodes=8192,threads=threads)
    t2=time.perf_counter_ns()
    return {"native_bounds_ms":(t1-t0)/1e6,
            "producer_and_verifier_ms":(t2-t1)/1e6,
            "full_ms":(t2-t0)/1e6,
            "final":result["final"],
            "bounds":tightening.copy(),
            "bound_max":np.max(tightening,axis=(0,1)).tolist(),
            "nodes":int(result["nodes"]),
            "leaves":int(len(result["leaves"]))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz",nargs="+",type=Path,required=True)
    p.add_argument("--pieces",nargs="+",type=int,default=[8,16,32])
    p.add_argument("--div-ulps",nargs="+",type=float,default=[2.0,4.0])
    p.add_argument("--sqrt-ulps",type=float,default=4.0)
    p.add_argument("--warmups",type=int,default=30)
    p.add_argument("--repeats",type=int,default=15)
    p.add_argument("--seed",type=int,default=20261117)
    p.add_argument("--threads",nargs="+",type=int,default=[1],
                   help="OpenMP threads for native bound, producer and verifier (identical results)")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if not a.npz or not a.pieces or any(x<1 or x>128 for x in a.pieces):
        p.error("pieces must be in [1,128]")
    if a.repeats<3 or a.warmups<0 or any(x<1 or x>8 for x in a.div_ulps):
        p.error("invalid repeat or FP32 division-ULP bound")
    configs=sorted(set((pieces,float(d),int(th)) for pieces in a.pieces for d in a.div_ulps for th in a.threads))
    rng=random.Random(a.seed)
    all_rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        builds=[K.build("k3_certifier.cpp",Path(tmp)/"p.so"),
                K.build("k3_verifier.cpp",Path(tmp)/"v.so")]
        P,V=K.Producer(Path(tmp)/"p.so"),K.Verifier(Path(tmp)/"v.so")
        for file in a.npz:
            blob=file.read_bytes()
            with np.load(file,allow_pickle=False) as z:
                arrays={k:z[k].copy() for k in z.files}
            before=hashlib.sha256(arrays["effective_geometry"].tobytes()+
                                  arrays["timing_controls"].tobytes()).hexdigest()
            results={key:[] for key in configs}
            # Balanced randomized configuration ordering per repeat.
            for _ in range(a.warmups):
                for pieces,d,th in configs:
                    one(P,V,arrays,pieces,d,a.sqrt_ulps,th)
            for repeat in range(a.repeats):
                order=configs[:]
                rng.shuffle(order)
                for pieces,d,th in order:
                    r=one(P,V,arrays,pieces,d,a.sqrt_ulps,th)
                    r["bounds_sha256"]=hashlib.sha256(r.pop("bounds").tobytes()).hexdigest()
                    results[(pieces,d,th)].append(r)
            after=hashlib.sha256(arrays["effective_geometry"].tobytes()+
                                 arrays["timing_controls"].tobytes()).hexdigest()
            if before!=after:
                raise RuntimeError("original frozen output mutated")
            for (pieces,d,th),reps in results.items():
                final={x["final"] for x in reps}
                if len(final)!=1:
                    raise RuntimeError("nondeterministic certificate for identical frozen inputs")
                all_rows.append({
                    "path_id":file.stem,
                    "source_npz_sha256":hashlib.sha256(blob).hexdigest(),
                    "source_output_sha256":before,
                    "pieces":pieces,"div_ulps":d,"sqrt_ulps":a.sqrt_ulps,"threads":th,
                    "bounds_sha256":reps[0]["bounds_sha256"],
                    "status":reps[0]["final"],"nodes":reps[0]["nodes"],
                    "leaves":reps[0]["leaves"],
                    "max_bound":reps[0]["bound_max"],
                    "native_bound_ms_p50":statistics.median(r["native_bounds_ms"] for r in reps),
                    "complete_ms_p50":statistics.median(r["full_ms"] for r in reps),
                    "raw":reps})
            print(f"{file.stem}: {len(configs)} configurations, original hash unchanged",flush=True)
    output={
        "scope":"FROZEN_OUTPUT_NATIVE_REPORTED_VALUES_CONDITIONAL_ON_FP32_ASSUMPTIONS",
        "warning":"This does not establish physical-time derivatives or native time-law consistency",
        "upstream_sha":"c301cb00a87012c2808b1a60649c4065c9b3c95a",
        "cpu":platform.platform(),"affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        "warmups":a.warmups,"repeats":a.repeats,"builds":builds,
        "configs":[{"pieces":pieces,"div_ulps":d,"threads":th} for pieces,d,th in configs],
        "rows":all_rows}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(f"Wrote {len(all_rows)} frozen output/configuration rows to {a.out}")


if __name__=="__main__":
    main()
