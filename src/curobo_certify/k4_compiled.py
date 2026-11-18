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
FLAGS = ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off", "-fopenmp", "-fPIC", "-shared", "-Wall", "-Wextra"]


ALLOWED_EXTRA = {"-O0", "-O1", "-O2", "-O3", "-march=native", "-g"}


def build(out: Path, extra=()) -> dict:
    """Only allowlisted extra flags (review F-P5-1: -ffinite-math-only, -fassociative-math,
    -ffp-contract=fast etc. would invalidate the outward-rounding argument)."""
    bad = [f for f in extra if f not in ALLOWED_EXTRA]
    if bad:
        raise ValueError(f"refusing compiler flags {bad}: not in the K4 allowlist")
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


# ------------------------------------------------------------------------------------------------------
# Compiled K4 path producer (tm1) - k4_certify_path in native/k4_rnea.cpp
# ------------------------------------------------------------------------------------------------------
STATUS = {0: "CERTIFIED", 1: "VIOLATED_TORQUE", 2: "VIOLATED_X", 3: "UNKNOWN_DEPTH", 4: "UNKNOWN_NODES",
          5: "INVALID_INPUT", 6: "NUMERICAL_ERROR"}


class _Leaf(C.Structure):
    _fields_ = [("span", C.c_int32), ("depth", C.c_int32), ("bits", C.c_uint32), ("pad", C.c_uint32),
                ("lo", C.c_double * 16), ("hi", C.c_double * 16)]


class _Wit(C.Structure):
    _fields_ = [("span", C.c_int32), ("joint", C.c_int32), ("depth", C.c_int32), ("where", C.c_int32),
                ("bits", C.c_uint32), ("pad", C.c_uint32), ("lo", C.c_double), ("hi", C.c_double),
                ("limit_lo", C.c_double), ("limit_hi", C.c_double)]


def bits_to_path(depth: int, bits: int) -> str:
    return "".join("R" if (bits >> (depth - 1 - k)) & 1 else "L" for k in range(depth))


def bytes_digest(geometry, timing, contract_sha256: str, lower, upper) -> str:
    """Input binding for the compiled pipeline: raw float32 bytes + contract SHA + limits (hex)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(geometry, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(np.asarray(timing).reshape(-1), dtype=np.float32).tobytes())
    h.update(contract_sha256.encode())
    for v in list(np.asarray(lower, float)) + list(np.asarray(upper, float)):
        h.update(float(v).hex().encode())
    return h.hexdigest()


class CompiledPath(CompiledCoefficients):
    def __init__(self, model, lib_path: Path):
        super().__init__(model, lib_path)
        L = self.lib
        if L.k4_leaf_size() != C.sizeof(_Leaf) or L.k4_witness_size() != C.sizeof(_Wit):
            raise RuntimeError("K4 struct layout mismatch")
        dp, ip, fp = C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_float)
        L.k4_certify_path.argtypes = [fp, C.c_int, C.c_int, fp, dp, dp, C.c_int, ip, ip, dp, dp, dp, C.c_int, C.c_int,
                                      C.POINTER(_Leaf), C.c_int, ip, C.POINTER(C.c_longlong), C.POINTER(_Wit), C.c_int]
        L.k4_certify_path.restype = C.c_int
        from .compiled_k3 import timing_intervals
        self.tlo_tab, self.thi_tab = timing_intervals()
        self.cap = 1 << 14
        self.buf = (_Leaf * self.cap)()

    def certify(self, geometry, timing, lower, upper, *, max_depth=12, max_nodes=65536, threads=1):
        g = np.ascontiguousarray(geometry, dtype=np.float32)
        tc = np.ascontiguousarray(np.asarray(timing).reshape(-1), dtype=np.float32)
        lo = np.ascontiguousarray(lower, dtype=np.float64)
        hi = np.ascontiguousarray(upper, dtype=np.float64)
        cnt, nodes, wit = C.c_int(), C.c_longlong(), _Wit()
        dp, ip, fp = C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_float)
        while True:
            st = self.lib.k4_certify_path(g.ctypes.data_as(fp), g.shape[0], g.shape[1], tc.ctypes.data_as(fp),
                                          self.tlo_tab.ctypes.data_as(dp), self.thi_tab.ctypes.data_as(dp),
                                          self.model.n, self.parent.ctypes.data_as(ip), self.joint.ctypes.data_as(ip),
                                          self.params.ctypes.data_as(dp), lo.ctypes.data_as(dp), hi.ctypes.data_as(dp),
                                          int(max_depth), int(max_nodes), self.buf, self.cap, C.byref(cnt),
                                          C.byref(nodes), C.byref(wit), int(threads))
            if st == 7:
                self.cap *= 4
                self.buf = (_Leaf * self.cap)()
                continue
            break
        d = self.dof
        leaves = [(int(lf.span), bits_to_path(int(lf.depth), int(lf.bits)), list(lf.lo[:d]), list(lf.hi[:d]))
                  for lf in self.buf[:cnt.value]]
        witness = None
        if st in (1, 2):
            from fractions import Fraction
            u = Fraction(2 * int(wit.bits) + int(wit.where), 2 ** (int(wit.depth) + 1))
            witness = {"span": int(wit.span), "joint": int(wit.joint), "u": str(u),
                       "torque_enclosure": [wit.lo, wit.hi], "limit": [wit.limit_lo, wit.limit_hi]}
        util = np.zeros(d)
        for (_s, _p, a, b) in leaves:
            a, b = np.array(a), np.array(b)
            util = np.maximum(util, np.maximum(np.where(b > 0, b / hi, 0), np.where(a < 0, a / lo, 0)))
        return {"status": STATUS.get(st, "NUMERICAL_ERROR"), "nodes": int(nodes.value), "leaves": leaves,
                "witness": witness, "worst_utilization": util.tolist(),
                "max_depth": max((len(p) for _s, p, _a, _b in leaves), default=0),
                "input_digest": bytes_digest(g, tc, self.model.contract_sha256, lo, hi)}


# ------------------------------------------------------------------------------------------------------
# Independent compiled verifier (native/k4_verify.cpp); parameters from the verifier's OWN contract parser
# ------------------------------------------------------------------------------------------------------
VSRC = Path(__file__).resolve().parent / "native" / "k4_verify.cpp"
VERIFY_CODES = {0: "PASS", 1: "PARTITION_INVALID", 2: "LEAF_NOT_VERIFIED", 3: "RECORDED_BOX_INVALID", 4: "INVALID_OR_NUMERICAL"}


def build_verifier(out: Path, extra=()) -> dict:
    bad = [f for f in extra if f not in ALLOWED_EXTRA]
    if bad:
        raise ValueError(f"refusing compiler flags {bad}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", *FLAGS, *extra, str(VSRC), "-o", str(out)], check=True)
    return {"source_sha256": hashlib.sha256(VSRC.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "flags": FLAGS + list(extra)}


class _VLeaf(C.Structure):
    _fields_ = _Leaf._fields_


class CompiledVerifier:
    def __init__(self, contract: dict, lib_path: Path, *, expected_contract_sha256: str | None = None):
        from .k4_verify import FROZEN_CONTRACT_V2_SHA256, WorldFrameDynamics, _contract_ok
        exp = expected_contract_sha256 or FROZEN_CONTRACT_V2_SHA256
        if not _contract_ok(contract, exp):
            raise ValueError("model contract hash mismatch")
        self.contract = contract
        W = WorldFrameDynamics(contract)
        rows = []
        for i in range(W.n):
            ivs = ([W.Rfix[i][r][s] for r in range(3) for s in range(3)] + list(W.p_off[i]) + [W.m[i]] + list(W.c[i])
                   + [W.I[i][r][s] for r in range(3) for s in range(3)])
            rows.extend(v for x in ivs for v in (float(np.asarray(x.lo)), float(np.asarray(x.hi))))
        rows.extend(v for x in W.g for v in (float(np.asarray(x.lo)), float(np.asarray(x.hi))))
        self.params = np.array(rows, dtype=np.float64)
        self.parent = np.array(W.parent, dtype=np.int32)
        self.joint = np.array(W.joint, dtype=np.int32)
        self.n, self.dof = W.n, W.dof
        self.lib = C.CDLL(str(lib_path))
        if self.lib.k4v_leaf_size() != C.sizeof(_VLeaf):
            raise RuntimeError("verifier leaf layout mismatch")
        dp, ip, fp = C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_float)
        self.lib.k4v_verify.argtypes = [fp, C.c_int, C.c_int, fp, dp, dp, C.c_int, ip, ip, dp, dp, dp,
                                        C.POINTER(_VLeaf), C.c_int, C.c_int, C.POINTER(C.c_longlong), C.c_int]
        self.lib.k4v_verify.restype = C.c_int
        from .compiled_k3 import timing_intervals
        self.tlo_tab, self.thi_tab = timing_intervals()

    def verify(self, geometry, timing, result: dict, lower, upper, *, refine=8, threads=1):
        g = np.ascontiguousarray(geometry, dtype=np.float32)
        tc = np.ascontiguousarray(np.asarray(timing).reshape(-1), dtype=np.float32)
        lo = np.ascontiguousarray(lower, dtype=np.float64)
        hi = np.ascontiguousarray(upper, dtype=np.float64)
        if result.get("status") != "CERTIFIED":
            return {"pass": False, "reason": "not a certificate"}
        if result.get("input_digest") != bytes_digest(g, tc, self.contract["contract_sha256"], lo, hi):
            return {"pass": False, "reason": "input digest missing or mismatched"}
        leaves = result["leaves"]
        arr = (_VLeaf * max(1, len(leaves)))()
        try:
            for i, (sp, path, blo, bhi) in enumerate(leaves):
                if any(ch not in "LR" for ch in path) or len(path) > 30:
                    return {"pass": False, "reason": "malformed leaf path"}
                arr[i].span, arr[i].depth = int(sp), len(path)
                arr[i].bits = int(path.replace("L", "0").replace("R", "1") or "0", 2)
                for j in range(self.dof):
                    arr[i].lo[j], arr[i].hi[j] = float(blo[j]), float(bhi[j])
        except (TypeError, ValueError, IndexError) as ex:
            return {"pass": False, "reason": f"malformed certificate: {ex}"}
        cnt = (C.c_longlong * 2)()
        dp, ip, fp = C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_float)
        rc = self.lib.k4v_verify(g.ctypes.data_as(fp), g.shape[0], g.shape[1], tc.ctypes.data_as(fp),
                                 self.tlo_tab.ctypes.data_as(dp), self.thi_tab.ctypes.data_as(dp), self.n,
                                 self.parent.ctypes.data_as(ip), self.joint.ctypes.data_as(ip),
                                 self.params.ctypes.data_as(dp), lo.ctypes.data_as(dp), hi.ctypes.data_as(dp),
                                 arr, len(leaves), int(refine), cnt, int(threads))
        return {"pass": rc == 0, "reason": VERIFY_CODES.get(rc, "UNKNOWN"), "boxes_verified": int(cnt[0]),
                "refinements": int(cnt[1])}
