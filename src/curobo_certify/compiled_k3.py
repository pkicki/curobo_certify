"""ctypes wrapper for the compiled complete K3 producer and independent verifier.

Producer: ``native/k3_certifier.cpp``; verifier: ``native/k3_verifier.cpp``.
A producer CERTIFIED result is only accepted as a certificate when the
independent verifier also passes (``certify_and_verify``).
"""
from __future__ import annotations

import ctypes as C
import hashlib
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(__file__).resolve().parent / "native"
# -fvisibility=hidden: only the k3c_*/k3v_* C API is exported. Exported internal
# helpers were found to be interposable by libc symbols (div) inside Python.
FLAGS = ("-std=c++17", "-O3", "-shared", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-fno-fast-math", "-ffp-contract=off",
         "-Wall", "-Wextra", "-Werror")
STATUS = {0: "CERTIFIED", 1: "VIOLATED", 2: "UNKNOWN_DEPTH", 3: "UNKNOWN_NODES", 4: "INVALID_INPUT",
          5: "UNKNOWN_BOUNDARY", 6: "VIOLATED_BOUNDARY", 7: "NUMERICAL_ERROR", 8: "UNKNOWN_UNIFORM"}
FAMILIES = ("positive_x", "velocity", "acceleration_plus", "acceleration_minus", "jerk")


class Leaf(C.Structure):
    _fields_ = [("span", C.c_int32), ("family", C.c_int32), ("joint", C.c_int32), ("depth", C.c_int32),
                ("bits", C.c_uint32), ("upper", C.c_double)]


class Witness(C.Structure):
    _fields_ = [("span", C.c_int32), ("family", C.c_int32), ("joint", C.c_int32), ("depth", C.c_int32),
                ("bits", C.c_uint32), ("where", C.c_int32), ("lo", C.c_double)]


def build(source: str, out: Path, extra=()) -> dict:
    src = NATIVE / source
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", *FLAGS, *extra, str(src), f"-Wl,--version-script={NATIVE / 'exports.map'}", "-o", str(out)]
    subprocess.run(cmd, check=True)
    return {"source": str(src.relative_to(ROOT)), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "exports_map_sha256": hashlib.sha256((NATIVE / "exports.map").read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "command": cmd,
            "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]}


@lru_cache(maxsize=1)
def timing_intervals() -> tuple[np.ndarray, np.ndarray]:
    """Outward binary64 intervals of exact (s-m/29)^p coefficients, [29][4 asc p][4 k]."""
    from .native_error_bound import exact_timing_table, fdn, fup
    t = exact_timing_table()  # [m][P=3-p][k]
    lo = np.empty((29, 4, 4)); hi = np.empty((29, 4, 4))
    for m in range(29):
        for p in range(4):
            for k in range(4):
                v = t[m][3 - p][k]
                lo[m, p, k], hi[m, p, k] = (float(v), float(v)) if Fraction(float(v)) == v else (fdn(v), fup(v))
    return np.ascontiguousarray(lo), np.ascontiguousarray(hi)


def _ptr(a, ctype):
    return a.ctypes.data_as(C.POINTER(ctype))


class Producer:
    def __init__(self, lib_path: Path):
        self.lib = C.CDLL(str(lib_path))
        L = self.lib
        assert L.k3c_leaf_size() == C.sizeof(Leaf) and L.k3c_witness_size() == C.sizeof(Witness)
        fp, dp, ip = C.POINTER(C.c_float), C.POINTER(C.c_double), C.POINTER(C.c_int)
        L.k3c_certify.argtypes = [fp, C.c_int, C.c_int, fp, fp, fp, dp, dp, dp, dp, dp, dp,
                                  C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, ip, C.POINTER(C.c_longlong), ip,
                                  C.POINTER(Leaf), C.c_int, ip, C.POINTER(Witness)]
        L.k3c_certify.restype = C.c_int
        L.k3c_common_spans.argtypes = [C.c_int, C.POINTER(C.c_longlong), C.POINTER(C.c_longlong), C.c_int]
        self.tlo, self.thi = timing_intervals()
        self.cap = 1 << 16
        self.buf = (Leaf * self.cap)()

    def common_spans(self, C_count: int):
        num = (C.c_longlong * (C_count + 40))(); den = (C.c_longlong * (C_count + 40))()
        ns = self.lib.k3c_common_spans(C_count, num, den, C_count + 40)
        return [(Fraction(num[i], den[i]), Fraction(num[i + 1], den[i + 1])) for i in range(ns)]

    def certify(self, geometry, timing, q_start, q_goal, limits, *, tighten=None, max_depth=12,
                max_nodes=65536, mode="adaptive", witnesses=True, strict_rest=True) -> dict:
        g = np.ascontiguousarray(geometry, dtype=np.float32)
        tc = np.ascontiguousarray(np.asarray(timing).reshape(-1), dtype=np.float32)
        qs = np.ascontiguousarray(q_start, dtype=np.float32).reshape(-1)
        qg = np.ascontiguousarray(q_goal, dtype=np.float32).reshape(-1)
        v, a, j = (np.ascontiguousarray(limits[k], dtype=np.float64) for k in ("velocity", "acceleration", "jerk"))
        tig = None if tighten is None else np.ascontiguousarray(tighten, dtype=np.float64)
        fam = (C.c_int * 5)(); nodes = C.c_longlong(); bstat = C.c_int(); count = C.c_int(); wit = Witness()
        while True:
            st = self.lib.k3c_certify(_ptr(g, C.c_float), g.shape[0], g.shape[1], _ptr(tc, C.c_float),
                                      _ptr(qs, C.c_float), _ptr(qg, C.c_float), _ptr(v, C.c_double), _ptr(a, C.c_double),
                                      _ptr(j, C.c_double), _ptr(self.tlo, C.c_double), _ptr(self.thi, C.c_double),
                                      None if tig is None else _ptr(tig, C.c_double),
                                      max_depth, max_nodes, 0 if mode == "adaptive" else 1, int(witnesses),
                                      int(strict_rest), fam, C.byref(nodes), C.byref(bstat), self.buf, self.cap,
                                      C.byref(count), C.byref(wit))
            if count.value <= self.cap:
                break
            self.cap = 1 << int(np.ceil(np.log2(count.value)))
            self.buf = (Leaf * self.cap)()
        leaves = np.ctypeslib.as_array(self.buf, shape=(self.cap,))[: count.value].copy()
        return {"status": STATUS[st], "families": {f: STATUS[fam[i]] for i, f in enumerate(FAMILIES)},
                "boundary": ("PASS", "VIOLATED", "UNKNOWN")[bstat.value], "nodes": nodes.value,
                "leaves": leaves, "witness": {k: getattr(wit, k) for k, _ in Witness._fields_} if st == 1 else None}


def input_digest(geometry, timing, q_start, q_goal, limits, tighten=None) -> str:
    """SHA-256 of the exact certificate inputs in the canonical byte layout the verifier hashes."""
    h = hashlib.sha256()
    g = np.ascontiguousarray(geometry, dtype=np.float32)
    h.update(np.array(g.shape, dtype="<i4").tobytes())
    for arr in (g, np.asarray(timing, dtype=np.float32), np.asarray(q_start, dtype=np.float32),
                np.asarray(q_goal, dtype=np.float32)):
        h.update(np.ascontiguousarray(arr, dtype="<f4").reshape(-1).tobytes())
    for k in ("velocity", "acceleration", "jerk"):
        h.update(np.ascontiguousarray(limits[k], dtype="<f8").reshape(-1).tobytes())
    if tighten is not None:
        h.update(np.ascontiguousarray(tighten, dtype="<f8").reshape(-1).tobytes())
    return h.hexdigest()


VERIFY_REASONS = {0: "PASS", 1: "invalid dimensions", 2: "input digest mismatch", 3: "nonfinite input",
                  10: "endpoint obligation not representable", 11: "negative endpoint squared speed",
                  12: "endpoint not finitely traversable", 13: "endpoint position not representable",
                  14: "endpoint position mismatch", 15: "strict-rest obligation not representable",
                  16: "nonzero endpoint velocity", 17: "nonzero endpoint acceleration",
                  20: "malformed leaf", 21: "residual without leaves", 22: "duplicate leaf", 23: "overlapping leaves",
                  24: "leaves do not cover [0,1]", 30: "nonpositive (tightened) limit", 31: "nonfinite residual",
                  40: "leaf hull exceeds zero", 41: "strict positivity not established on leaf",
                  90: "rounding mode is not nearest", 91: "FTZ/DAZ enabled"}


class Verifier:
    def __init__(self, lib_path: Path):
        self.lib = C.CDLL(str(lib_path))
        fp, dp = C.POINTER(C.c_float), C.POINTER(C.c_double)
        self.lib.k3v_verify.argtypes = [fp, C.c_int, C.c_int, fp, fp, fp, dp, dp, dp, dp, C.c_int, C.c_char_p,
                                        C.POINTER(Leaf), C.c_int, C.POINTER(C.c_int)]
        self.lib.k3v_verify.restype = C.c_int

    def verify(self, geometry, timing, q_start, q_goal, limits, leaves: np.ndarray, digest: str, *,
               tighten=None, strict_rest=True) -> dict:
        g = np.ascontiguousarray(geometry, dtype=np.float32)
        tc = np.ascontiguousarray(np.asarray(timing).reshape(-1), dtype=np.float32)
        qs = np.ascontiguousarray(q_start, dtype=np.float32).reshape(-1)
        qg = np.ascontiguousarray(q_goal, dtype=np.float32).reshape(-1)
        v, a, j = (np.ascontiguousarray(limits[k], dtype=np.float64) for k in ("velocity", "acceleration", "jerk"))
        tig = None if tighten is None else np.ascontiguousarray(tighten, dtype=np.float64)
        lv = np.ascontiguousarray(leaves)
        if lv.dtype != np.dtype(Leaf):
            lv = lv.astype(np.dtype(Leaf))
        detail = C.c_int()
        code = self.lib.k3v_verify(_ptr(g, C.c_float), g.shape[0], g.shape[1], _ptr(tc, C.c_float), _ptr(qs, C.c_float),
                                   _ptr(qg, C.c_float), _ptr(v, C.c_double), _ptr(a, C.c_double), _ptr(j, C.c_double),
                                   None if tig is None else _ptr(tig, C.c_double), int(strict_rest),
                                   digest.encode(), lv.ctypes.data_as(C.POINTER(Leaf)), len(lv), C.byref(detail))
        return {"pass": code == 0, "code": code, "reason": VERIFY_REASONS.get(code, "unknown"), "residual": detail.value}


def certify_and_verify(producer: Producer, verifier: Verifier, geometry, timing, q_start, q_goal, limits, *,
                       tighten=None, **opts) -> dict:
    """Producer + independent verifier. CERTIFIED only if both agree."""
    r = producer.certify(geometry, timing, q_start, q_goal, limits, tighten=tighten, **opts)
    r["digest"] = input_digest(geometry, timing, q_start, q_goal, limits, tighten)
    if r["status"] == "CERTIFIED":
        r["verification"] = verifier.verify(geometry, timing, q_start, q_goal, limits, r["leaves"], r["digest"],
                                            tighten=tighten, strict_rest=opts.get("strict_rest", True))
        r["final"] = "CERTIFIED" if r["verification"]["pass"] else "REJECTED_BY_VERIFIER"
    else:
        r["verification"] = None
        r["final"] = r["status"]
        # A witness against TIGHTENED limits (native error bound subtracted) is not a
        # violation of the original limits by the native output: fail to UNKNOWN.
        if tighten is not None and r["status"] == "VIOLATED":
            r["final"] = "UNKNOWN_TIGHTENED_WITNESS"
    return r



@lru_cache(maxsize=1)
def _immutable_native_table() -> np.ndarray:
    """Avoid NPZ-like disk reads in every fresh-request native bound evaluation.

    Cached bytes are read-only. The pinned source-table digest remains the
    provenance authority and must be checked when starting the run.
    """
    table = np.ascontiguousarray(
        np.load(ROOT / "fixtures/timeopt_timing_polynomial_c301cb.npy"),
        dtype=np.float32)
    table.setflags(write=False)
    return table


def native_bounds(producer: Producer, geometry, timing, *, pieces=16, delta=1e-6, div_ulps=2.0, sqrt_ulps=3.0,
                  native_table=None) -> np.ndarray:
    """Compiled rigorous native FP32 error bounds [ns, D, 3] (velocity, acceleration, jerk)."""
    L = producer.lib
    L.k3c_native_bounds.argtypes = [C.POINTER(C.c_float), C.c_int, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_float),
                                    C.POINTER(C.c_double), C.POINTER(C.c_double), C.c_int, C.c_double, C.c_double,
                                    C.c_double, C.POINTER(C.c_double)]
    L.k3c_native_bounds.restype = C.c_int
    g = np.ascontiguousarray(geometry, dtype=np.float32)
    tc = np.ascontiguousarray(np.asarray(timing).reshape(-1), dtype=np.float32)
    table = (_immutable_native_table() if native_table is None
             else np.ascontiguousarray(native_table, dtype=np.float32))
    ns = len(producer.common_spans(g.shape[0]))
    out = np.zeros((ns, g.shape[1], 3))
    got = L.k3c_native_bounds(_ptr(g, C.c_float), g.shape[0], g.shape[1], _ptr(tc, C.c_float), _ptr(table, C.c_float),
                              _ptr(producer.tlo, C.c_double), _ptr(producer.thi, C.c_double), pieces, delta,
                              div_ulps, sqrt_ulps, _ptr(out, C.c_double))
    if got != ns:
        raise RuntimeError(f"native bound computation failed: code {got}")
    return out
