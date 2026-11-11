"""ctypes adapter for the *partial* compiled exact-model interval Bernstein hull.

NO complete K3 certification: input model's boundaries, regularity, strict
positivity, interior stationary speed, native CUDA evaluation and replay are
not validated by this component. An all-negative hull is only a necessary
component of certification. It is never promoted to CERTIFIED_K3.
"""
from __future__ import annotations

import ctypes
import json
import math
from fractions import Fraction
from pathlib import Path

FAMILIES = ('positive_x', 'velocity', 'acceleration_plus',
            'acceleration_minus', 'jerk')


def rational_enclosure(value) -> tuple[float, float]:
    """Smallest binary64 interval enclosing an exact rational (finite only)."""
    exact = Fraction(value)
    try:
        nearest = float(exact)
    except (OverflowError, ValueError) as exc:
        raise ValueError('rational outside finite double range') from exc
    if not math.isfinite(nearest):
        raise ValueError('rational outside finite double range')
    nearest_exact = Fraction.from_float(nearest)
    lo = math.nextafter(nearest, -math.inf) if nearest_exact > exact else nearest
    hi = math.nextafter(nearest, math.inf) if nearest_exact < exact else nearest
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError('rational cannot be safely enclosed by finite doubles')
    if not (Fraction.from_float(lo) <= exact <= Fraction.from_float(hi)):
        raise AssertionError('failed to enclose exact rational')
    return lo, hi


def _pad_poly(values):
    if not 1 <= len(values) <= 4:
        raise ValueError('only degree<=3 nonempty polynomials are supported')
    return tuple(values) + (0,) * (4-len(values))


def pack_model(model):
    if not isinstance(model, dict) or model.get('schema_version') != 1 or model.get('basis') != 'local_power_ascending_u':
        raise ValueError('unsupported mathematical spline model format')
    spans = model.get('spans')
    limits = model.get('limits')
    if not isinstance(spans, list) or not spans or not isinstance(limits, dict):
        raise ValueError('missing spans or limits')
    n = len(spans)
    d = len(spans[0]['q'])
    if d <= 0:
        raise ValueError('empty joint dimension')
    q, x, h = [], [], []
    for span in spans:
        if len(span['q']) != d:
            raise ValueError('joint dimension mismatch')
        left, right = Fraction(span['left']), Fraction(span['right'])
        if right <= left:
            raise ValueError('nonpositive interval width')
        h.append(right-left)
        for joint in span['q']:
            q.extend(_pad_poly(joint))
        x.extend(_pad_poly(span['x']))
    names = ('velocity','acceleration','jerk')
    vals = []
    for name in names:
        sequence = limits[name]
        if len(sequence) != d:
            raise ValueError('joint limit dimension mismatch')
        v = [Fraction(value) for value in sequence]
        if any(value <= 0 for value in v):
            raise ValueError('all limits must be positive')
        vals.append(v)
    arrays=[]
    for seq in (q,x,h,*vals):
        bound_pairs = [rational_enclosure(v) for v in seq]
        arrays.extend([[z[0] for z in bound_pairs], [z[1] for z in bound_pairs]])
    # Arguments: qlo/hi,xlo/hi,hlo/hi,vlo/hi,alo/hi,jlo/hi
    return n,d,arrays


class CompiledHull:
    """Loads an explicit compiled .so; never builds or downloads it implicitly."""
    def __init__(self, library: str | Path):
        self.library = str(Path(library).resolve())
        self._lib = ctypes.CDLL(self.library)
        self._fn = self._lib.timeopt_k3_hull_v1
        pd = ctypes.POINTER(ctypes.c_double)
        self._fn.argtypes = [pd]*12+[ctypes.c_int,ctypes.c_int,pd,pd]
        self._fn.restype = ctypes.c_int
        self._coeff_fn = self._lib.timeopt_k3_coefficients_v1
        self._coeff_fn.argtypes = [pd]*12+[ctypes.c_int,ctypes.c_int,pd,pd,pd,pd]
        self._coeff_fn.restype = ctypes.c_int

    def evaluate_packed(self, n, d, arrays):
        if len(arrays) != 12:
            raise ValueError('invalid array schema')
        buffers=[]
        for values in arrays:
            ctype=ctypes.c_double*len(values)
            buffers.append(ctype(*values))
        count=n*d*len(FAMILIES)
        output_l=(ctypes.c_double*count)()
        output_h=(ctypes.c_double*count)()
        code=self._fn(*buffers,n,d,output_l,output_h)
        if code != 0:
            raise ValueError(f'compiled interval hull rejected inputs or environment: status {code}')
        results=[]
        for i in range(count):
            lo,hi=output_l[i],output_h[i]
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError('invalid compiled bound')
            results.append((lo,hi))
        return results

    def evaluate(self, model):
        n,d,arrays=pack_model(model)
        return self.evaluate_packed(n,d,arrays)

    def evaluate_coefficients(self, model):
        """Return every padded-degree Bernstein coefficient enclosure.

        Family slot offsets within each (span,joint) row are
        positive_x=0, velocity=4, acceleration+=12, acceleration-=17, jerk=22.
        Families use degrees 3,7,4,4,9 respectively; output storage is padded
        to 32 slots for a stable C ABI. This validates hull arithmetic only.
        """
        n,d,arrays=pack_model(model)
        buffers=[]
        for values in arrays:
            ctype=ctypes.c_double*len(values)
            buffers.append(ctype(*values))
        count=n*d*len(FAMILIES)
        hull_l=(ctypes.c_double*count)();hull_h=(ctypes.c_double*count)()
        coeff_count=n*d*32
        coeff_l=(ctypes.c_double*coeff_count)();coeff_h=(ctypes.c_double*coeff_count)()
        code=self._coeff_fn(*buffers,n,d,hull_l,hull_h,coeff_l,coeff_h)
        if code != 0:
            raise ValueError(f'compiled coefficient enclosure rejected inputs or environment: status {code}')
        coeffs=[]
        for lo,hi in zip(coeff_l,coeff_h):
            if not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:
                raise ValueError('invalid coefficient enclosure')
            coeffs.append((lo,hi))
        return coeffs


def partial_summary(model, bounds):
    """Report per-family *initial-hull* status; never full certified status."""
    n=len(model['spans']);d=len(model['spans'][0]['q'])
    if len(bounds)!=n*d*len(FAMILIES):
        raise ValueError('bound count mismatch')
    family={name:{'upper':-math.inf,'count':0,'hull_passes':0}
            for name in FAMILIES}
    for s in range(n):
        for j in range(d):
            for f,name in enumerate(FAMILIES):
                if f==0 and j!=0:  # one positivity residual per common span
                    continue
                lo,hi=bounds[(s*d+j)*5+f]
                family[name]['upper']=max(family[name]['upper'],hi)
                family[name]['count']+=1
                family[name]['hull_passes']+=int(hi<=0)
    return {'status':'INITIAL_HULLS_NONPOSITIVE_PARTIAL' if all(
            item['hull_passes']==item['count'] for item in family.values())
            else 'INITIAL_HULL_UNKNOWN_OR_VIOLATED',
            'scope':'EXPLORATORY_COMPILED_BINARY64_INITIAL_HULL_ONLY',
            'native_parity':'NOT_EVALUATED',
            'proof_replay':'NOT_EVALUATED',
            'boundary_and_regularity':'NOT_EVALUATED',
            'full_k3_certificate':False,
            'family':family}


def read_model(blob:bytes):
    return json.loads(blob)
