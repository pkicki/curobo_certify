#!/usr/bin/env python3
"""Exhaustive native time->progress audit against the rigorous exact time law.

For each frozen native output:
 1. CPU: rigorous enclosure of the exact time law T(s) on 2^L dyadic cells
    (curobo_certify.time_law_grid) -> tables T_lo, T_hi, fmin, fmax.
 2. GPU: for EVERY float32 requested time t in [0, T_native] (all bit patterns),
    run the pinned upstream progress_at_time (compiled with cuRobo's NVRTC options)
    and compute a certified upper bound on |s_native(t) - s*(t)|, where s* is the
    exact inverse of the mathematical time law (held at 1 after the model duration).
 3. Transfer to physical-time state deviations: |q(s_n)-q(s*)| <= max|q'| eps_s,
    |v(s_n)-v(s*)| <= max|dv/ds| eps_s, |a(s_n)-a(s*)| <= max|da/ds| eps_s, with
    rigorous Bernstein-hull Lipschitz bounds; jerk is piecewise (one-sided at knots)
    and gets max|dj/ds| eps_s plus the largest jerk jump across a geometry knot.
Scope: NATIVE_TIME_INVERSION_EXHAUSTIVE_FP32_TIMES. The native s(t) values are
observed on the GPU (no arithmetic assumption needed for them); the claim rests on
the harness executing the same compiled upstream code as production (checked by
bitwise reproduction of saved production samples) and on the rigorous CPU bracket.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from curobo_certify import time_law_grid as TG  # noqa: E402
from curobo_certify.boundary_reference import derivative  # noqa: E402
from curobo_certify.exact_k3_reference import bernstein_coeffs, poly_add, poly_mul, poly_scale  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402
from native_same_s_falsify import native_inputs, run_progress  # noqa: E402

AUDIT = ROOT / "src/curobo_certify/native/time_inverse_audit.cuh"


def f32bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def fup(v) -> float:
    return float(np.nextafter(float(v), np.inf))


def lipschitz(spans):
    """Rigorous per-joint upper bounds over the whole path (Bernstein hulls of exact polynomials)."""
    D = len(spans[0].q)
    out = {k: [Fraction(0)] * D for k in ("q", "v", "a", "j", "jump_j")}
    xmin, xmax, dx = None, Fraction(0), Fraction(0)
    for sp in spans:
        h = sp.right - sp.left
        b = bernstein_coeffs(sp.x)
        xmin = min(b) if xmin is None else min(xmin, min(b))
        xmax = max(xmax, max(b))
        dx = max(dx, max(abs(c) for c in bernstein_coeffs(derivative(sp.x, 1, h))))
    if xmin <= 0:
        raise ValueError("x not certified positive by hull")
    sx_lo = Fraction(int((xmin * 10**12) ** 0.5 * 0.999999), 10**6)  # rational lower bound of sqrt(xmin)
    if not sx_lo * sx_lo <= xmin:
        raise ValueError("sqrt lower bound failed")  # explicit check (asserts vanish under -O)
    sx_hi = Fraction(int((xmax * 10**12) ** 0.5 * 1.000001) + 1, 10**6)
    if not sx_hi * sx_hi >= xmax:
        raise ValueError("sqrt upper bound failed")
    prev_j = None
    for sp in spans:
        h = sp.right - sp.left
        x, x1, x2 = sp.x, derivative(sp.x, 1, h), derivative(sp.x, 2, h)
        x3 = derivative(sp.x, 3, h)
        jvals = []
        for j, q in enumerate(sp.q):
            q1, q2, q3 = (derivative(q, o, h) for o in (1, 2, 3))
            M = lambda p: max(abs(c) for c in bernstein_coeffs(p))  # noqa: E731
            out["q"][j] = max(out["q"][j], M(q1))
            # dv/ds = q'' sqrt(x) + q' x' /(2 sqrt(x))
            out["v"][j] = max(out["v"][j], M(q2) * sx_hi + M(q1) * M(x1) / (2 * sx_lo))
            # da/ds = q''' x + 1.5 q'' x' + 0.5 q' x''  (polynomial)
            ell = poly_add(poly_add(poly_mul(q3, x), poly_scale(poly_mul(q2, x1), Fraction(3, 2))),
                           poly_scale(poly_mul(q1, x2), Fraction(1, 2)))
            out["a"][j] = max(out["a"][j], M(ell))
            # j = sqrt(x) ell ; dj/ds = ell x'/(2 sqrt x) + sqrt(x) ell',  ell' = 2.5 q''' x' + 2 q'' x'' + 0.5 q' x'''
            ell1 = poly_add(poly_add(poly_scale(poly_mul(q3, x1), Fraction(5, 2)), poly_scale(poly_mul(q2, x2), 2)),
                            poly_scale(poly_mul(q1, x3), Fraction(1, 2)))
            out["j"][j] = max(out["j"][j], M(ell) * M(x1) / (2 * sx_lo) + sx_hi * M(ell1))
            jvals.append((q3[0] if q3 else Fraction(0)))
        if prev_j is not None:
            for j in range(D):  # jerk jump at a geometry knot: |Δq'''| * x^{3/2} <= |Δq'''| xmax^{3/2}
                out["jump_j"][j] = max(out["jump_j"][j], abs(jvals[j] - prev_j[j]) * sx_hi ** 3)
        prev_j = jvals
    return {k: [fup(v) for v in vs] for k, vs in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--log2-cells", type=int, default=22)
    ap.add_argument("--chunk", type=int, default=1 << 26)
    ap.add_argument("--check-samples", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from native_same_s_falsify import Harness
    dev = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(dev)

    class AuditHarness(Harness):
        def __init__(self, device):
            import native_same_s_falsify as NS
            orig = NS.HARNESS
            super().__init__(device)
            from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
            cfg = TrajectoryKernelCfg()
            sources = [cfg.kernel_dir / "timeopt" / "timeopt_common.cuh",
                       cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh", orig, AUDIT]
            self.source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
            for short in ("inverse_audit", "reported_audit"):
                name = "curobo_certify_harness::" + short
                self.kernels[short] = self.cache.get_or_compile_kernel(
                    source_files=sources, kernel_name=name, include_dirs=cfg.get_include_dirs(), compile_flags=self.flags)
                self.qualified[short] = name
    h = AuditHarness(dev)
    rows = []
    for path in a.npz:
        t0 = time.time()
        raw = path.read_bytes()
        with np.load(path, allow_pickle=False) as z:
            arr = {k: z[k].copy() for k in z.files}
        grid = TG.build_grid(arr["timing_controls"], a.log2_cells)
        row = {"path": path.stem, "source_npz_sha256": hashlib.sha256(raw).hexdigest(), "grid_status": grid.status}
        if grid.status != "BOUNDED_EXACT_MODEL":
            rows.append(row); continue
        inp = native_inputs(arr, torch, dev)
        tab = {k: torch.as_tensor(np.ascontiguousarray(v), device=dev) for k, v in
               (("T_lo", grid.T_lo), ("T_hi", grid.T_hi), ("fmin", grid.fmin), ("fmax", grid.fmax))}
        Tn = float(arr["native_timing_time_grid"][-1])
        last = f32bits(Tn)
        res = {k: torch.zeros(1, dtype=torch.int64, device=dev) for k in ("any", "ahead", "behind")}
        start, total = 0, last + 1
        while start < total:
            cnt = min(a.chunk, total - start)
            h.launch("inverse_audit", cnt, inp["polynomial"], inp["timing"], inp["nodes"], inp["weights"], inp["time"],
                     np.int32(inp["intervals"]), np.uint32(start), np.uint32(cnt), tab["T_lo"], tab["T_hi"],
                     tab["fmin"], tab["fmax"], np.int32(a.log2_cells), res["any"], res["ahead"], res["behind"])
            start += cnt

        # Exhaustive reported-value audit over the same float32 times (and the tail is zero by construction).
        lim64 = {n: torch.as_tensor(np.minimum(np.abs(arr[k][0]), np.abs(arr[k][1])).astype(np.float64), device=dev)
                 for n, k in (("v", "velocity_limits"), ("a", "acceleration_limits"), ("j", "jerk_limits"))}
        mr = torch.zeros(3, dtype=torch.int64, device=dev); nf = torch.zeros(1, dtype=torch.int64, device=dev)
        start = 0
        while start < total:
            cnt = min(a.chunk, total - start)
            h.launch("reported_audit", cnt, inp["controls"], inp["start"], inp["goal"], inp["timing"], inp["polynomial"],
                     inp["nodes"], inp["weights"], inp["time"], np.int32(inp["intervals"]), np.int32(inp["dof"]),
                     np.int32(inp["knots"]), np.int32(1), np.uint32(start), np.uint32(cnt),
                     lim64["v"], lim64["a"], lim64["j"], mr, nf)
            start += cnt

        def unpack(t):
            v = int(t.item()) & 0xFFFFFFFFFFFFFFFF
            e = struct.unpack("<f", struct.pack("<I", v >> 32))[0]
            tt = struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0]
            return float(e), float(tt)
        reported = {fam: dict(zip(("max_utilization", "at_time"), unpack(mr[i:i + 1])))
                    for i, fam in enumerate(("velocity", "acceleration", "jerk"))}
        reported_nonfinite = int(nf.item())
        eps, t_at = unpack(res["any"])
        ahead, t_ahead = unpack(res["ahead"])
        behind, t_behind = unpack(res["behind"])
        # Sanity cross-check of the audit on random float32 times (CPU bracket, harness s).
        rng = np.random.default_rng(20261117)
        ts = (rng.random(a.check_samples) * Tn).astype(np.float32)
        _, s_nat = run_progress(h, inp, ts)
        cpu_max = 0.0
        for t, s in zip(ts, s_nat):
            lo, hi = TG.inverse_bracket(grid, float(t))
            cpu_max = max(cpu_max, float(s) - hi, lo - float(s))
        spans = extract_timeopt_spans(arr["effective_geometry"], arr["timing_controls"])
        L = lipschitz(spans)
        dev_bounds = {k: [fup(eps * v) for v in L[k]] for k in ("q", "v", "a", "j")}
        dev_bounds["j_with_knot_jump"] = [fup(d + jj) for d, jj in zip(dev_bounds["j"], L["jump_j"])]
        lim = {n: np.minimum(np.abs(arr[k][0]), np.abs(arr[k][1])).astype(float)
               for n, k in (("v", "velocity_limits"), ("a", "acceleration_limits"), ("j", "jerk_limits"))}
        row.update({"times_enumerated": total, "native_duration_s": Tn, "model_duration_bracket_s": list(grid.duration),
                    "grid_cells": grid.n_cells, "grid_summation_error_s": grid.summation_error,
                    "eps_s_certified_max": eps, "at_time": t_at, "ahead_max": ahead, "ahead_at": t_ahead,
                    "behind_max": behind, "behind_at": t_behind,
                    "cpu_crosscheck_samples": a.check_samples, "cpu_crosscheck_max": cpu_max,
                    "crosscheck_consistent": cpu_max <= eps * (1 + 1e-6) + 1e-30,
                    "lipschitz": L, "state_deviation_bounds": dev_bounds,
                    "exhaustive_reported": reported, "exhaustive_reported_nonfinite": reported_nonfinite,
                    "exhaustive_reported_status": ("CERTIFIED_EXHAUSTIVE" if reported_nonfinite == 0 and
                                                   all(r["max_utilization"] <= 1.0 for r in reported.values())
                                                   else "VIOLATED_OR_NONFINITE"),
                    "relative_deviation_vs_limits": {k: max(d / l for d, l in zip(dev_bounds[k], lim[k])) for k in ("v", "a", "j")},
                    "knot_window_time_s": fup(eps / float(np.sqrt(np.float64(min(arr["timing_controls"].min(), 1e300))))) if arr["timing_controls"].min() > 0 else None,
                    "elapsed_s": time.time() - t0})
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("path", "times_enumerated", "eps_s_certified_max", "at_time", "ahead_max",
                                              "behind_max", "cpu_crosscheck_max", "crosscheck_consistent",
                                              "relative_deviation_vs_limits", "exhaustive_reported_status", "exhaustive_reported", "elapsed_s")}), flush=True)
    out = {"scope": "NATIVE_TIME_INVERSION_EXHAUSTIVE_FP32_TIMES", "gpu": gpu, "compile_flags": h.flags,
           "harness_and_upstream_source_sha256": h.source_hashes,
           "grid_module_sha256": hashlib.sha256((ROOT / "src/curobo_certify/time_law_grid.py").read_bytes()).hexdigest(),
           "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
