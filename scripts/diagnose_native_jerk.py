#!/usr/bin/env python3
"""Stage-wise native TimeOpt jerk comparison on retained Panda pilot arrays.

This is a source-faithful FP32 *reproduction*, not native kernel instrumentation:
the arrays contain native endpoint outputs, but not kernel scratch registers. The
script evaluates the pinned factored B-spline equations and timing equations in
explicit float32 operation order, then compares them with adapter FP64 and the
independent FP64 reference. It labels that limitation in every output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


F = np.float32


def fadd(a, b): return F(F(a) + F(b))
def fsub(a, b): return F(F(a) - F(b))
def fmul(a, b): return F(F(a) * F(b))
def fdiv(a, b): return F(F(a) / F(b))


def path_fp32(points, s):
    spans = points.shape[0] - 3
    coord = fmul(F(s), F(spans))
    seg = min(int(coord), spans - 1)
    u = fsub(coord, F(seg))
    cp = points[seg:seg + 4].astype(np.float32, copy=True)
    centered = np.asarray([fsub(cp[k], cp[0]) for k in range(4)], dtype=np.float32)
    om = fsub(1.0, u)
    u2 = fmul(u, u)
    one6 = F(1.0 / 6.0)
    b1 = np.array([
        fmul(F(-0.5), fmul(om, om)),
        fmul(F(0.5), fsub(fmul(F(3), u2), fmul(F(4), u))),
        fmul(F(0.5), fsub(fadd(F(1), fmul(F(2), u)), fmul(F(3), u2))),
        fmul(F(0.5), u2)], dtype=np.float32)
    b2 = np.array([om, fsub(fmul(F(3), u), F(2)), fsub(F(1), fmul(F(3), u)), u], dtype=np.float32)
    b3 = np.array([-1, 3, -3, 1], dtype=np.float32)
    # The native context dot-products centered controls in source order.
    d1 = np.zeros(points.shape[1], dtype=np.float32)
    d2 = np.zeros_like(d1)
    d3 = np.zeros_like(d1)
    pos = np.zeros_like(d1)
    b0 = np.array([
        fmul(one6, fmul(fmul(om, om), om)),
        fmul(one6, fadd(fsub(F(4), fmul(F(6), u2)), fmul(F(3), fmul(u2, u)))),
        fmul(one6, fsub(fadd(fadd(F(1), fmul(F(3), u)), fmul(F(3), u2)), fmul(F(3), fmul(u2, u)))),
        fmul(one6, fmul(u2, u))], dtype=np.float32)
    for k in range(4):
        pos = np.asarray([fadd(pos[j], fmul(cp[k, j], b0[k])) for j in range(len(pos))])
        d1 = np.asarray([fadd(d1[j], fmul(centered[k, j], b1[k])) for j in range(len(d1))])
        d2 = np.asarray([fadd(d2[j], fmul(centered[k, j], b2[k])) for j in range(len(d2))])
        d3 = np.asarray([fadd(d3[j], fmul(centered[k, j], b3[k])) for j in range(len(d3))])
    d1 = np.asarray([fmul(v, F(spans)) for v in d1])
    d2 = np.asarray([fmul(v, F(spans * spans)) for v in d2])
    d3 = np.asarray([fmul(v, F(spans * spans * spans)) for v in d3])
    return pos, d1, d2, d3, seg, u


def timing_replay(controls, times, duration, t):
    """Pinned timing_state/progress_at_time equations, all intermediate ops rounded FP32."""
    nctrl = len(controls)
    breaks = np.linspace(0.0, 1.0, nctrl - 2)
    knots = np.r_[np.zeros(3), breaks, np.ones(3)]
    spline = BSpline(knots, np.eye(nctrl), 3)
    lefts = breaks[:-1]
    # Native static polynomial order is [cubic, quadratic, linear, constant].
    poly = np.asarray(np.stack([spline(lefts, nu=k) / (1, 1, 2, 6)[k]
                                for k in (3, 2, 1, 0)], axis=1), dtype=np.float32)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    nodes = np.asarray(0.5 * (nodes + 1), dtype=np.float32)
    weights = np.asarray(0.5 * weights, dtype=np.float32)

    def state(s):
        span = min(28, max(0, int(fmul(s, F(29)))))
        left = fdiv(F(span), F(29))
        local = fsub(s, left)
        c = np.zeros(4, np.float32)
        for power in range(4):
            for k in range(4):
                c[power] = fadd(c[power], fmul(poly[span, power, span + k], controls[span + k]))
        x = fadd(fmul(fadd(fmul(fadd(fmul(c[0], local), c[1]), local), c[2]), local), c[3])
        x1 = fadd(fmul(fadd(fmul(F(3), fmul(c[0], local)), fmul(F(2), c[1])), local), c[2])
        x2 = fadd(fmul(F(6), fmul(c[0], local)), fmul(F(2), c[1]))
        return x, x1, x2

    t = F(t); duration = F(duration)
    cell = min(len(times) - 2, max(0, int(np.searchsorted(times, t, side="right") - 1)))
    left, right = fdiv(F(cell), F(len(times) - 1)), fdiv(F(cell + 1), F(len(times) - 1))
    dt = max(float(fsub(times[cell + 1], times[cell])), 1e-20)
    s = fadd(left, fdiv(fmul(fsub(right, left), fsub(t, times[cell])), F(dt)))
    for _ in range(5):
        elapsed = F(0)
        for k in range(6):
            sk = fadd(left, fmul(fsub(s, left), nodes[k]))
            xk, _, _ = state(sk)
            elapsed = fadd(elapsed, fmul(F(1 / np.sqrt(max(float(xk), 1e-20))), weights[k]))
        x, _, _ = state(s)
        residual = fsub(fadd(times[cell], fmul(fsub(s, left), elapsed)), t)
        s = F(max(float(left), min(float(right), float(fsub(s, fmul(residual, F(np.sqrt(max(float(x), 1e-20)))))))))
    if t <= 0: s = F(0)
    elif t >= duration: s = F(1)
    return float(s), tuple(float(v) for v in state(s)), poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, default=Path("runs/native_pilot_resolved"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/native_jerk_diagnostic"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"evidence_kind": "native_output_vs_source_faithful_fp32_reproduction",
               "campaign_measurements": False,
               "native_intermediate_limitation": "TimeOpt kernels expose no saved scratch values; FP32 intermediate columns are source-faithful replay, not instrumented device registers.",
               "fma": "CUDA compilation may contract multiply-adds; NumPy replay rounds each operation separately and cannot establish device FMA behavior.",
               "paths": []}
    for meta_path in sorted(a.pilot_dir.glob("pilot-panda-*.json")):
        meta = json.loads(meta_path.read_text())
        if not meta.get("planner_success"):
            continue
        arrays_path = Path(meta["arrays_path"])
        if not arrays_path.is_absolute(): arrays_path = Path.cwd() / arrays_path
        with np.load(arrays_path, allow_pickle=False) as z:
            geom = z["effective_geometry"].copy()
            times = z["parity_times"].astype(np.float32)
            native = z["parity_native"].astype(np.float32)
            ref = z["parity_reference"].astype(np.float64)
            timing = z["timing_controls"].astype(np.float32)
            time_grid = z["native_timing_time_grid"].astype(np.float32)
            duration = float(z["native_duration"][0])
        n = len(times)
        qf = np.empty((n, geom.shape[1]), np.float32)
        q1 = qf.copy(); q2 = qf.copy(); q3 = qf.copy()
        segs = np.empty(n, np.int32); us = np.empty(n, np.float32)
        timing_states, poly = [], None
        for i in range(n):
            si, xi, poly = timing_replay(timing, time_grid, duration, times[i])
            timing_states.append((si, *xi))
        timing_states = np.asarray(timing_states, dtype=np.float64)
        s, x, x1, x2 = timing_states.T
        for i in range(n):
            qf[i], q1[i], q2[i], q3[i], segs[i], us[i] = path_fp32(geom, s[i])
        # Adapter FP64 reference is retained independently; native-source FP32 uses its own x and derivatives.
        terms64 = np.stack([ref[3] * x[:, None], 1.5 * ref[2] * x1[:, None], 0.5 * ref[1] * x2[:, None]])
        terms32 = np.empty((3, n, geom.shape[1]), dtype=np.float32)
        for i in range(n):
            terms32[0, i] = np.asarray([fmul(q3[i, j], F(x[i])) for j in range(geom.shape[1])])
            terms32[1, i] = np.asarray([fmul(fmul(F(1.5), q2[i, j]), F(x1[i])) for j in range(geom.shape[1])])
            terms32[2, i] = np.asarray([fmul(fmul(F(0.5), q1[i, j]), F(x2[i])) for j in range(geom.shape[1])])
        total32 = np.asarray([[fadd(fadd(terms32[0, i, j], terms32[1, i, j]), terms32[2, i, j])
                               for j in range(geom.shape[1])] for i in range(n)], dtype=np.float32)
        model32geom = np.asarray([[fmul(F(np.sqrt(max(float(x[i]), 0))), total32[i, j])
                                   for j in range(geom.shape[1])] for i in range(n)], dtype=np.float32)
        model64 = np.sqrt(x)[:, None] * terms64.sum(axis=0)
        # Quantify native basis replay versus adapter path derivatives.
        error = native[3] - ref[6]
        stage = {"fp32_source_reproduction_vs_fp64_adapter_jerk_max_abs": float(np.max(np.abs(model32geom - ref[6]))),
                 "native_vs_adapter_jerk_max_abs": float(np.max(np.abs(error))),
                 "fp32_source_reproduction_vs_native_jerk_max_abs": float(np.max(np.abs(model32geom - native[3]))),
                 "max_segment": int(segs.max()), "points": n}
        imax, jmax = np.unravel_index(np.argmax(np.abs(error)), error.shape)
        nearest_knot = round(s[imax] * (geom.shape[0] - 3)) / (geom.shape[0] - 3)
        stage["max_scaled_relative_native_adapter"] = float(np.max(
            np.abs(error) / np.maximum(1.0, np.abs(ref[6]))))
        stage["worst_sample"] = {
            "time_s": float(times[imax]), "path_progress": float(s[imax]),
            "joint": int(jmax), "nearest_path_knot": float(nearest_knot),
            "distance_to_path_knot": float(abs(s[imax] - nearest_knot)),
            "native_jerk": float(native[3, imax, jmax]),
            "adapter_fp64_jerk": float(ref[6, imax, jmax]),
            "source_fp32_replay_jerk": float(model32geom[imax, jmax]),
            "replay_minus_adapter": float(model32geom[imax, jmax] - ref[6, imax, jmax]),
            "native_minus_replay": float(native[3, imax, jmax] - model32geom[imax, jmax]),
            "equation_terms_adapter_geometry": [float(terms64[k, imax, jmax]) for k in range(3)],
        }
        summary["paths"].append({"path_id": meta["source_id"], "stage_metrics": stage})
        data = {"time_s": times, "path_progress": s, "segment": segs, "local_coordinate": us,
                "native_jerk": native[3], "adapter_fp64_jerk": ref[6],
                "reproduced_fp32_source_jerk": model32geom,
                "error_native_minus_adapter": error,
                "term_qsss_x": terms64[0], "term_1p5_qss_x1": terms64[1], "term_0p5_qs_x2": terms64[2]}
        np.savez_compressed(a.out_dir / f"{meta['source_id']}.npz", **data)
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        for j in range(native.shape[-1]):
            axes[0].plot(times, native[3, :, j], alpha=.55, lw=.8, label=f"native j{j}")
            axes[0].plot(times, ref[6, :, j], alpha=.6, lw=.7, ls="--")
        axes[0].set_ylabel("jerk [rad/s³]"); axes[0].set_title(f"{meta['source_id']}: solid native, dashed adapter FP64")
        axes[1].plot(times, np.max(np.abs(error), axis=1)); axes[1].set_ylabel("max joint |error|")
        axes[2].plot(times, np.max(np.abs(model32geom - native[3]), axis=1), label="FP32 source replay - native")
        axes[2].plot(times, np.max(np.abs(model32geom - ref[6]), axis=1), label="FP32 source replay - adapter")
        axes[2].set_ylabel("max absolute error"); axes[2].set_xlabel("physical time [s]"); axes[2].legend()
        for ax in axes:
            for k in np.arange(1/29, 1, 1/29):
                ax.axvline(np.interp(k, s, times), color="gray", alpha=.12, lw=.5)
            ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(a.out_dir / f"{meta['source_id']}.png", dpi=140); plt.close(fig)
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"paths": len(summary["paths"]), "out_dir": str(a.out_dir),
                      "evidence_kind": summary["evidence_kind"]}, indent=2))


if __name__ == "__main__": main()
