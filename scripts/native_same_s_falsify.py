#!/usr/bin/env python3
"""Falsification tests of the rigorous native-evaluation error bound.

Runs in the pinned cuRobo environment on the GPU. For each frozen native path:

1. Compile a harness with cuRobo's own NVRTC path and options, appended to the
   UNMODIFIED pinned headers timeopt_common.cuh + timeopt_path_kernel.cuh.
2. Bitwise reproduction gate: harness ``progress_eval`` at the production
   requested times must reproduce the saved production outputs (S8193 and
   event-aware S32768 diagnostics) bit for bit; ``same_s_eval`` at the 1,176
   captured device progress values must reproduce captured device jerk.
3. Same-progress sweep: dense/knot-adjacent/random float32 s values; compare
   the native outputs with the exact model (evaluated in binary64 at the native
   geometry point) and check |native-model| <= rigorous bound + binary64 slack.

A pass is evidence that the bound is not contradicted on these finite sets. It
is NOT itself a proof; the proof is the bound derivation in
``curobo_certify.native_error_bound`` under its stated arithmetic assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import native_error_bound as NEB  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402

HARNESS = ROOT / "src/curobo_certify/native/same_s_harness.cuh"
TABLE = ROOT / "fixtures/timeopt_timing_polynomial_c301cb.npy"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Harness:
    def __init__(self, device):
        import torch
        from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg  # type: ignore
        from curobo._src.context import get_runtime  # type: ignore
        self.torch = torch
        self.device = device
        cfg = TrajectoryKernelCfg()
        self.cache = get_runtime().get_cuda_core_cache()
        self.flags = cfg.get_compile_flags()
        sources = [cfg.kernel_dir / "timeopt" / "timeopt_common.cuh",
                   cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh", HARNESS]
        self.source_hashes = {p.name: sha(p) for p in sources}
        self.kernels = {name: self.cache.get_or_compile_kernel(
            source_files=sources, kernel_name="curobo_certify_harness::" + name, include_dirs=cfg.get_include_dirs(),
            compile_flags=self.flags) for name in ("progress_eval", "same_s_eval")}
        self.qualified = {n: "curobo_certify_harness::" + n for n in self.kernels}

    def launch(self, name, n, *args):
        from cuda.core import LaunchConfig  # type: ignore
        from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel  # type: ignore
        stream = self.cache.get_stream_wrapper(self.torch.cuda.current_stream(self.device))
        launch_kernel(self.qualified[name], stream, LaunchConfig(grid=(n + 255) // 256, block=256), self.kernels[name],
                      *[x.data_ptr() if isinstance(x, self.torch.Tensor) else x for x in args])
        self.torch.cuda.synchronize(self.device)


def native_inputs(arrays, torch, device):
    eff = arrays["effective_geometry"].astype(np.float32)
    C, dof = eff.shape
    knots = C - 5
    if not (np.array_equal(eff[-1], eff[-2]) and np.array_equal(eff[-1], eff[-3])):
        raise ValueError("effective geometry does not end with three goal controls")
    start = np.stack([eff[0], eff[0], eff[1], eff[2]]).astype(np.float32)  # start[0] unused
    controls = np.concatenate([eff[3:3 + knots - 1], eff[-1:]], 0).astype(np.float32)
    goal = eff[-1:].astype(np.float32)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)  # noqa: E731
    return dict(controls=t(controls), start=t(start), goal=t(goal),
                timing=t(arrays["timing_controls"].astype(np.float32)),
                polynomial=t(np.load(TABLE).astype(np.float32)),
                nodes=t((0.5 * (nodes + 1)).astype(np.float32)),
                weights=t((0.5 * weights).astype(np.float32)),
                time=t(arrays["native_timing_time_grid"].astype(np.float32)),
                knots=knots, dof=dof, intervals=len(arrays["native_timing_time_grid"]) - 1)


def run_progress(h, inp, times):
    torch = h.torch
    req = torch.as_tensor(times.astype(np.float32), device=h.device)
    n = len(times)
    out = torch.empty((4, n, inp["dof"]), dtype=torch.float32, device=h.device)
    prog = torch.empty(n, dtype=torch.float32, device=h.device)
    h.launch("progress_eval", n, inp["controls"], inp["start"], inp["goal"], inp["timing"],
             inp["polynomial"], inp["nodes"], inp["weights"], inp["time"], req, out, prog,
             np.int32(n), np.int32(inp["dof"]), np.int32(inp["knots"]), np.int32(inp["intervals"]),
             np.int32(1))
    return out.cpu().numpy(), prog.cpu().numpy()


def run_same_s(h, inp, s_values):
    torch = h.torch
    sv = torch.as_tensor(s_values.astype(np.float32), device=h.device)
    n = len(s_values)
    out = torch.empty((4, n, inp["dof"]), dtype=torch.float32, device=h.device)
    h.launch("same_s_eval", n, inp["controls"], inp["start"], inp["goal"], inp["timing"],
             inp["polynomial"], sv, out, np.int32(n), np.int32(inp["dof"]), np.int32(inp["knots"]),
             np.int32(1))
    return out.cpu().numpy()


def model_reference(spans_exact, eff, timing_controls, s32):
    """Exact model (binary64 evaluation) at the native geometry point s_g."""
    S = eff.shape[0] - 3
    coord = (s32.astype(np.float32) * np.float32(S)).astype(np.float64)
    seg = np.minimum(np.floor(coord).astype(np.int64), S - 1)
    t = coord - seg
    e = eff.astype(np.float64)
    P = np.stack([e[seg + r] for r in range(4)], 0)        # [4, n, dof]
    c = P - P[0:1]
    t2 = t * t
    vb = np.stack([-3 * (1 - t) ** 2 / 6, (9 * t2 - 12 * t) / 6, (-9 * t2 + 6 * t + 3) / 6, 3 * t2 / 6])
    ab = np.stack([6 * (1 - t) / 6, (18 * t - 12) / 6, (-18 * t + 6) / 6, 6 * t / 6])
    jb = np.array([-1.0, 3.0, -3.0, 1.0])[:, None] * np.ones_like(t)[None]
    q1 = np.einsum("knd,kn->nd", c, vb) * S
    q2 = np.einsum("knd,kn->nd", c, ab) * S ** 2
    q3 = np.einsum("knd,kn->nd", c, jb) * S ** 3
    sg = (seg + t) / S
    table = NEB.exact_timing_table()
    coeffs = np.array([[float(sum((table[m][P_][k] * Fraction(float(timing_controls[m + k]))
                                    for k in range(4)), Fraction(0))) for P_ in range(4)]
                       for m in range(29)])
    m = np.minimum(np.floor(sg * 29).astype(np.int64), 28)
    loc = sg - m / 29.0
    a0, a1, a2, a3 = (coeffs[m, i] for i in range(4))
    x0 = ((a0 * loc + a1) * loc + a2) * loc + a3
    x1 = (3 * a0 * loc + 2 * a1) * loc + a2
    x2 = 6 * a0 * loc + 2 * a1
    speed = np.sqrt(np.maximum(x0, 0))[:, None]
    vel = q1 * speed
    acc = q2 * x0[:, None] + 0.5 * q1 * x1[:, None]
    inner_abs = np.abs(q3 * x0[:, None]) + np.abs(1.5 * q2 * x1[:, None]) + np.abs(0.5 * q1 * x2[:, None])
    jerk = speed * (q3 * x0[:, None] + 1.5 * q2 * x1[:, None] + 0.5 * q1 * x2[:, None])
    slack = 1e-9 * (1 + speed * inner_abs + np.abs(q2 * x0[:, None]) + np.abs(q1 * x1[:, None]))
    lefts = np.array([float(sp.left) for sp in spans_exact])
    span_idx = np.clip(np.searchsorted(lefts, sg, side="right") - 1, 0, len(lefts) - 1)
    return {"velocity": vel, "acceleration": acc, "jerk": jerk}, slack, span_idx


def sweep_sets(eff, rng, dense: int, random_n: int):
    S = eff.shape[0] - 3
    knots_s = sorted(set([i / S for i in range(S + 1)] + [i / 29 for i in range(30)]))
    adj = []
    for k in knots_s:
        f = np.float32(k)
        v = f
        for _ in range(64):
            v = np.nextafter(v, np.float32(-1)); adj.append(v)
        v = f
        adj.append(f)
        for _ in range(64):
            v = np.nextafter(v, np.float32(2)); adj.append(v)
    adj = np.array(adj, dtype=np.float32)
    adj = adj[(adj >= 0) & (adj <= np.float32(1) + np.float32(4 * 2 ** -24))]
    grid = np.linspace(0, 1, dense, dtype=np.float64).astype(np.float32)
    rnd = rng.random(random_n).astype(np.float32)
    ends = np.concatenate([np.logspace(-30, -1, 4096), 1 - np.logspace(-7.2, -1, 4096)]).astype(np.float32)
    return {"knot_adjacent_pm64ulp": adj, "uniform_grid": grid, "random_uniform": rnd, "endpoint_logspaced": ends}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--traces", type=Path, default=ROOT / "runs/compiled_hull_panda/device_operands_joined.json")
    ap.add_argument("--dense", type=int, default=1 << 21)
    ap.add_argument("--random", type=int, default=1 << 20)
    ap.add_argument("--pieces", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    h = Harness(device)
    model = NEB.ArithmeticModel()
    traces = json.loads(args.traces.read_text())["records"] if args.traces.exists() else []
    rows = []
    for path in args.npz:
        t0 = time.time()
        raw_sha = sha(path)
        with np.load(path, allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        eff = arrays["effective_geometry"]
        tc = arrays["timing_controls"]
        spans = extract_timeopt_spans(eff, tc)
        bounds = NEB.common_span_bounds(model, spans, eff.tolist(), tc.tolist(), np.load(TABLE),
                                        pieces=args.pieces)
        E = {fam: np.array([[getattr(b, fam) for b in row] for row in bounds])
             for fam in ("velocity", "acceleration", "jerk")}
        inp = native_inputs(arrays, torch, device)
        row = {"path": path.stem, "source_npz_sha256": raw_sha, "gpu": gpu, "common_spans": len(spans)}
        # Gate 1: bitwise reproduction of production outputs at requested times.
        repro = {}
        for key in ("diagnostic_8193", "diagnostic_32768"):
            times = arrays[f"{key}_times"]
            out, prog = run_progress(h, inp, times)
            ref = arrays[f"{key}_states"]
            mism = int(np.sum(out.view(np.uint32) != ref.astype(np.float32).view(np.uint32)))
            repro[key] = {"samples": int(len(times)), "values": int(ref.size), "bit_mismatches": mism}
        row["production_bitwise_reproduction"] = repro
        # Gate 2: captured device jerk at captured progress.
        tr = [r for r in traces if r.get("path_id") == path.stem and r.get("source_arrays_sha256") == raw_sha]
        if tr:
            out = run_same_s(h, inp, np.array([r["s"] for r in tr], dtype=np.float32))
            got = out[3, np.arange(len(tr)), [r["joint"] for r in tr]]
            want = np.array([r["native_jerk"] for r in tr], dtype=np.float32)
            row["captured_trace_jerk"] = {"rows": len(tr), "bit_mismatches": int(np.sum(got.view(np.uint32) != want.view(np.uint32)))}
        # Gate 3: same-progress sweep against the bound.
        rng = np.random.default_rng(args.seed)
        sweep = {}
        for name, svals in sweep_sets(eff, rng, args.dense, args.random).items():
            out = run_same_s(h, inp, svals)
            ref, slack, sidx = model_reference(spans, eff, tc, svals)
            fam_rows = {}
            for k, fam in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
                err = np.abs(out[k].astype(np.float64) - ref[fam])
                bound = E[fam][sidx]
                # neighbouring-span bound for points within 1e-12 of a span boundary (binary64 ambiguity)
                bound = np.maximum(bound, E[fam][np.minimum(sidx + 1, len(spans) - 1)] *
                                   (np.abs(np.array([float(sp.right) for sp in spans])[sidx] - svals) < 1e-12)[:, None])
                ratio = err / (bound + slack)
                fam_rows[fam] = {"max_abs_error": float(err.max()), "max_ratio_to_bound": float(ratio.max()),
                                 "exceedances": int(np.sum(err > bound + slack)),
                                 "max_bound": float(bound.max())}
            sweep[name] = {"points": int(len(svals)), **fam_rows}
        row["same_progress_sweep"] = sweep
        row["elapsed_s"] = time.time() - t0
        rows.append(row)
        print(json.dumps({"path": row["path"], "repro": repro, "trace": row.get("captured_trace_jerk"),
                          "max_ratio": {n: max(v[f]["max_ratio_to_bound"] for f in ("velocity", "acceleration", "jerk"))
                                        for n, v in sweep.items()},
                          "exceed": sum(v[f]["exceedances"] for v in sweep.values() for f in ("velocity", "acceleration", "jerk"))}))
    result = {"scope": "FALSIFICATION_DIAGNOSTIC_NOT_PROOF", "gpu": gpu, "arithmetic_model": model.describe(),
              "compile_flags": h.flags, "harness_and_upstream_source_sha256": h.source_hashes,
              "timing_table_sha256": sha(TABLE), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
