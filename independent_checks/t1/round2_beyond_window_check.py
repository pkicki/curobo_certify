#!/usr/bin/env python3
"""R0 round 2 Q4: beyond-duration window bit range of scripts/native_time_audit_v2.py (lines 193-196)
recomputed with the v2 grid on all 224 audited paths. Run with the GPU-env python (numpy 2.x, as the driver):
  CUDA_VISIBLE_DEVICES="" conda run -n curobo-timeopt-c301cb python independent_checks/t1/round2_beyond_window_check.py
driver_dead_branch_paths: paths where numpy-2 float32 comparison makes the nextafter branch dead, so
stop_bits = RN32(T_exact_hi) < T_exact_hi (sound: no float32 lies in between; t >= T_exact_hi has s*=s=1)."""
import sys, json, glob, struct, hashlib, numpy as np
sys.path.insert(0,'src')
from curobo_certify import time_law_grid as TG
from fractions import Fraction as Q
print('numpy', np.__version__)
y = float(np.nextafter(np.float32(0.3), np.float32(1))) - 1e-12   # just below a float32
print('np.float32(y) >= y :', bool(np.float32(y) >= y), ' exact:', Q(float(np.float32(y))) >= Q(y))
f32b=lambda v: struct.unpack('<I',struct.pack('<f',v))[0]
b2f=lambda b: struct.unpack('<f',struct.pack('<I',b))[0]
rows=json.load(open('runs/margins/time_inversion_audit_224paths.json'))['rows']
m={}
for f in glob.glob('runs/**/*.npz',recursive=True): m.setdefault(f.split('/')[-1][:-4],[]).append(f)
stats={'paths':0,'window_paths':0,'stop_below_That':0,'max_window':0,'driver_dead_branch_paths':0}
for r in rows:
    f=[p for p in m[r['path']] if hashlib.sha256(open(p,'rb').read()).hexdigest()==r['source_npz_sha256']][0]
    z=np.load(f); Tn=float(z['native_timing_time_grid'][-1]); g=TG.build_grid(z['timing_controls'],22)
    Th=float(g.T_hi[-1]); last=f32b(Tn)
    drv = f32b(float(np.float32(Th))) if np.float32(Th) >= Th else f32b(float(np.nextafter(np.float32(Th), np.float32(np.inf))))
    exact_first = f32b(float(np.float32(Th))) if Q(float(np.float32(Th)))>=Q(Th) else f32b(float(np.float32(Th)))+1
    stats['paths']+=1
    if drv!=exact_first: stats['driver_dead_branch_paths']+=1
    if Q(b2f(drv)) < Q(Th): stats['stop_below_That']+=1
    w=max(0,drv-last); stats['max_window']=max(stats['max_window'],w); stats['window_paths']+= w>0
print(stats)
