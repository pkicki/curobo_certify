"""Compiled K4 coefficient backend (native/k4_rnea.cpp) for GeometryCache.

Build: g++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off -fPIC -shared (no FMA contraction, IEEE
round-to-nearest; the library refuses -ffast-math at compile time). Model parameters are packed as the
EXACT interval enclosures held by PandaModel (contract v2). The compiled backend must agree with the
Python reference in the differential test (tests/test_k4_compiled.py): its enclosures must contain the
Python point enclosures and be at least as tight up to rounding (ulps), never looser by orders.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from .k4_interval import IA

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent / "native" / "k4_rnea.cpp"
FLAGS = ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared", "-Wall", "-Wextra"]


def build(out: Path, extra=()) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", *FLAGS, *extra, str(SRC), "-o", str(out)]
    subprocess.run(cmd, check=True)
    return {"source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "flags": FLAGS + list(extra),
            "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]}


def _pack(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for i in range(model.n):
        ivs = ([model.Rf[i][r][s] for r in range(3) for s in range(3)] + list(model.pf[i]) + [model.mass[i]]
               + list(model.com[i]) + [model.Ic[i][r][s] for r in range(3) for s in range(3)])
        rows.extend(v for x in ivs for v in (float(np.asarray(x.lo)), float(np.asarray(x.hi))))
    rows.extend(v for x in model.gravity for v in (float(np.asarray(x.lo)), float(np.asarray(x.hi))))
    parent = np.array(model.parent, dtype=np.int32)
    parent[0] = 0
    return np.array(rows, dtype=np.float64), parent, np.array(model.joint, dtype=np.int32)


class CompiledCoefficients:
    def __init__(self, model, lib_path: Path):
        self.lib = C.CDLL(str(lib_path))
        if self.lib.k4_rnea_abi_version() != 1:
            raise RuntimeError("k4_rnea ABI mismatch")
        dp, ip = C.POINTER(C.c_double), C.POINTER(C.c_int)
        for fn in (self.lib.k4_box_coefficients, self.lib.k4_tm1_slopes):
            fn.argtypes = [C.c_int, C.c_int, ip, ip, dp, C.c_int, dp, dp]
            fn.restype = C.c_int
        self.model = model
        self.params, self.parent, self.joint = _pack(model)
        self.dof = model.dof

    def _call(self, fn, blocks, nbox):
        d = self.dof
        x = np.empty((nbox, len(blocks) * d, 2))
        for b, blk in enumerate(blocks):
            for j in range(d):
                x[:, b * d + j, 0] = np.broadcast_to(blk[j].lo, (nbox,))
                x[:, b * d + j, 1] = np.broadcast_to(blk[j].hi, (nbox,))
        x = np.ascontiguousarray(x)
        out = np.empty((nbox, 3 * d, 2))
        dp = C.POINTER(C.c_double)
        rc = fn(self.model.n, d, self.parent.ctypes.data_as(C.POINTER(C.c_int)),
                self.joint.ctypes.data_as(C.POINTER(C.c_int)), self.params.ctypes.data_as(dp), nbox,
                x.ctypes.data_as(dp), out.ctypes.data_as(dp))
        if rc != 0:
            raise RuntimeError("k4_rnea failed")
        return [[IA(out[:, k * d + j, 0], out[:, k * d + j, 1]) for j in range(d)] for k in range(3)]

    def coefficients(self, q, qp, qpp):
        n = int(np.asarray(q[0].lo).size)
        A, B, G = self._call(self.lib.k4_box_coefficients, [q, qp, qpp], n)
        return A, B, G

    def slopes(self, q, qp, qpp, qppp):
        n = int(np.asarray(q[0].lo).size)
        return self._call(self.lib.k4_tm1_slopes, [q, qp, qpp, qppp], n)
