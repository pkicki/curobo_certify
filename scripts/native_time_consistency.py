#!/usr/bin/env python3
"""Identical-requested-time comparison: native s_n(t) versus exact inverse s*(t).

The native progress s_n(t) is obtained from the harness ``progress_eval`` (which
must reproduce the production outputs bit for bit, gate checked here again).
The exact model time law is T(s) = int_0^s dsigma / sqrt(x(sigma)). T is
computed by composite 8-point Gauss-Legendre on 64 cells per timing span in
binary64 and cross-checked against 32 cells; s*(t) by safeguarded Newton.

Scope: HIGH_PRECISION_NUMERICAL_DIAGNOSTIC. It quantifies time-inversion
inconsistency and decomposes native-vs-model differences at identical t into
(a) same-progress evaluation error and (b) progress error. It is not a rigorous
continuous-time bound on s_n(t) - s*(t).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from native_same_s_falsify import Harness, model_reference, native_inputs, run_progress  # noqa: E402
from curobo_certify import native_error_bound as NEB  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


class TimeLaw:
    def __init__(self, timing_controls, cells=64):
        table = NEB.exact_timing_table()
        self.coeffs = np.array([[float(sum((table[m][P][k] * Fraction(float(timing_controls[m + k]))
                                             for k in range(4)), Fraction(0))) for P in range(4)]
                                for m in range(29)])
        g, w = np.polynomial.legendre.leggauss(8)
        self.g, self.w, self.cells = 0.5 * (g + 1), 0.5 * w, cells
        self.cum = np.zeros(30)
        for m in range(29):
            self.cum[m + 1] = self.cum[m] + self._span_integral(m, 1 / 29)

    def x(self, m, loc):
        a0, a1, a2, a3 = self.coeffs[m]
        return ((a0 * loc + a1) * loc + a2) * loc + a3

    def _span_integral(self, m, upto, cells=None):
        cells = cells or self.cells
        edges = np.linspace(0.0, upto, cells + 1)
        h = np.diff(edges)
        pts = edges[:-1, None] + h[:, None] * self.g[None]
        return float(np.sum(h[:, None] * self.w[None] / np.sqrt(self.x(m, pts))))

    def T(self, s, cells=None):
        s = min(max(float(s), 0.0), 1.0)
        m = min(int(s * 29), 28)
        return self.cum[m] + self._span_integral(m, s - m / 29, cells)

    def inverse(self, t):
        lo, hi = 0.0, 1.0
        s = min(max(t / self.cum[-1], 0.0), 1.0)
        for _ in range(60):
            f = self.T(s) - t
            if f > 0:
                hi = min(hi, s)
            else:
                lo = max(lo, s)
            if f == 0:
                return s
            m = min(int(s * 29), 28)
            step = f * np.sqrt(max(self.x(m, s - m / 29), 1e-300))
            if abs(step) < 1e-17:
                return s
            nxt = s - step
            s = nxt if lo <= nxt <= hi else 0.5 * (lo + hi)
        return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--max-times", type=int, default=8193)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    device = torch.device("cuda:0")
    h = Harness(device)
    rows = []
    for path in args.npz:
        with np.load(path, allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        eff, tc = arrays["effective_geometry"], arrays["timing_controls"]
        spans = extract_timeopt_spans(eff, tc)
        inp = native_inputs(arrays, torch, device)
        times = arrays["diagnostic_8193_times"][: args.max_times]
        out, s_native = run_progress(h, inp, times)
        ref_states = arrays["diagnostic_8193_states"][:, : args.max_times]
        mism = int(np.sum(out.view(np.uint32) != ref_states.astype(np.float32).view(np.uint32)))
        law = TimeLaw(tc)
        law32 = TimeLaw(tc, cells=32)
        quad_check = max(abs(law.T(s) - law32.T(s)) for s in np.linspace(0, 1, 97))
        duration_model = law.cum[-1]
        s_star = np.array([law.inverse(float(t)) for t in times])
        ds = s_native.astype(np.float64) - s_star
        ref_n, _, _ = model_reference(spans, eff, tc, s_native)          # model at native s
        ref_star, _, _ = model_reference(spans, eff, tc, s_star.astype(np.float32))
        fam = {}
        for k, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
            total = np.abs(out[k].astype(np.float64) - ref_star[name])
            same_s = np.abs(out[k].astype(np.float64) - ref_n[name])
            inversion = np.abs(ref_n[name] - ref_star[name])
            fam[name] = {"max_total_abs": float(total.max()), "max_same_progress_abs": float(same_s.max()),
                         "max_inversion_induced_abs": float(inversion.max())}
        row = {"path": path.stem, "source_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "times": int(len(times)), "production_bit_mismatches": mism,
               "native_duration_s": float(arrays["native_duration"][0]), "model_duration_s": float(duration_model),
               "duration_rel_diff": float((arrays["native_duration"][0] - duration_model) / duration_model),
               "quadrature_32_vs_64_cell_max_abs_s": float(quad_check),
               "progress_error": {"max_abs": float(np.abs(ds).max()), "median_abs": float(np.median(np.abs(ds))),
                                  "max_abs_in_float32_ulps_of_s": float(np.max(np.abs(ds) / np.maximum(np.spacing(np.abs(s_star).astype(np.float32)).astype(np.float64), 1e-45)))},
               "families": fam}
        rows.append(row)
        print(json.dumps({"path": row["path"], "mism": mism, "dur_rel": row["duration_rel_diff"],
                          "ds_max": row["progress_error"]["max_abs"], "ulps": row["progress_error"]["max_abs_in_float32_ulps_of_s"],
                          "jerk": fam["jerk"]}))
    res = {"scope": "HIGH_PRECISION_NUMERICAL_DIAGNOSTIC_NOT_PROOF", "gpu": torch.cuda.get_device_name(device),
           "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
