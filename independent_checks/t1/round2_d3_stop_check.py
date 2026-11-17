import numpy as np, struct
from fractions import Fraction as Q
print(np.__version__)
b2f=lambda b: struct.unpack('<f',struct.pack('<I',b))[0]
def d3(T):
    f=float(np.float32(T))
    if f < T: f=float(np.nextafter(np.float32(f), np.float32(np.inf)))
    return f
def ref(T):  # smallest float32 >= T by bisection on bit patterns (exact Fraction compare)
    lo,hi=0,0x7F800000
    while lo<hi:
        m=(lo+hi)//2
        if Q(b2f(m))>=Q(T): hi=m
        else: lo=m+1
    return b2f(lo)
rng=np.random.default_rng(0); bad=0; n=0
vals=list(np.exp(rng.uniform(np.log(1e-45),np.log(3e38),200000)))
vals+= [b2f(b) for b in rng.integers(1,0x7F7FFFFF,20000)]
vals+= [float(np.nextafter(b2f(b),0)) for b in rng.integers(2,0x7F7FFFFF,20000)]+[float(np.nextafter(b2f(b),1e39)) for b in rng.integers(1,0x7F7FFFFE,20000)]
vals+= [0.29515284505, 2.0**-149, 2.0**-150*1.0000001, 3.4028234663852886e38]
for T in vals:
    T=float(T); n+=1
    if d3(T)!=ref(T): bad+=1
print('checked',n,'mismatches',bad)
