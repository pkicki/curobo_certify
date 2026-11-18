#!/usr/bin/env python3
"""K4 HARD CASES (2): sampling-layout blind spots on TOPP-RA parametrizations.

TOPP-RA's output is constant path acceleration per grid interval, so its time law is exact in closed form:
on [s_i, s_{i+1}] with sd_i, sdd_i = (sd_{i+1}^2 - sd_i^2) / (2 ds):  s(tau) = s_i + sd_i tau + sdd_i tau^2 / 2.
For every certified TOPP-RA record (k4_toppra_certify) this script evaluates exact-model torque (float64, compiled
interval RNEA midpoints) on:
  uniform-TIME grids at rates 1 kHz, 10 kHz, 100 kHz, 1 MHz (a controller-rate / time-sampled checker), and
  uniform-s grids with N = 512, 4096, 32768, 262144,
and the time the trajectory spends above the limit (measured on a 2^20-point uniform-s scan mapped through the time
law; DIAGNOSTIC). A violation is MISSED by a sampler when the rigorous lb > 1 (K4 VIOLATED, verified witness) and the
sampler's max utilization <= 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RATES = (1e3, 1e4, 1e5, 1e6)
NS = (512, 4096, 32768, 262144)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toppra", type=Path, required=True)
    ap.add_argument("--cert", type=Path, nargs="+", required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--grids", nargs="+", default=["auto", "100"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from curobo_certify import k4_compiled as KC
    from curobo_certify import k4_sampling as KS
    from curobo_certify.k4_panda_model import PandaModel
    from curobo_certify.k4_toppra import toppra_spans, trimmed
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    c = json.loads(a.contract.read_text())
    if a.payload:
        from curobo_certify.k4_payload import with_payload
        c = with_payload(c, a.payload)
    M = PandaModel(c)
    lo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
    hi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
    tmp = tempfile.TemporaryDirectory()
    KC.build(Path(tmp.name) / "p.so")
    CC = KC.CompiledCoefficients(M, Path(tmp.name) / "p.so")
    tp = json.loads(a.toppra.read_text())
    cert = {(r["path"], r["grid"]): r for f in a.cert for r in json.loads(f.read_text())["rows"]}
    rows = []
    for rec in tp["rows"]:
        if rec["grid"] not in a.grids or not rec.get("ok"):
            continue
        cr = cert.get((rec["path"], rec["grid"]))
        if cr is None or cr["K4"] not in ("VIOLATED_TORQUE", "CERTIFIED"):
            continue
        with np.load(a.npz_dir / f"{rec['path']}.npz") as z:
            base = extract_timeopt_spans(z["effective_geometry"], z["timing_controls"])
        if tp.get("trim"):
            base, _ = trimmed(base)
        gs = np.array([float.fromhex(h) for h in rec["gridpoints_hex"]]); gs[0], gs[-1] = 0.0, 1.0
        sd = np.array([float.fromhex(h) for h in rec["sd_hex"]])
        grid = [Q(float(x)) for x in gs]
        spans = toppra_spans(base, grid, [Q(float(v)) ** 2 for v in sd])
        T = KS.SpanTable(spans)
        ds = np.diff(gs)
        dt = 2 * ds / np.maximum(sd[:-1] + sd[1:], 1e-300)
        tk = np.r_[0.0, np.cumsum(dt)]
        sdd = (sd[1:] ** 2 - sd[:-1] ** 2) / (2 * ds)

        def s_of_t(t):
            i = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, len(ds) - 1)
            tau = t - tk[i]
            return np.clip(gs[i] + sd[i] * tau + 0.5 * sdd[i] * tau * tau, gs[i], gs[i + 1])

        def U(s):
            return KS.utilization(KS.sampled_torque(CC, T, s), lo, hi).max(1)
        res = {"time": {}, "s": {}}
        dur = float(tk[-1])
        for r_ in RATES:
            n = int(np.ceil(dur * r_)) + 1
            if n > 2_000_000:
                continue
            res["time"][f"{int(r_)}Hz"] = {"n": n, "max_util": float(U(s_of_t(np.linspace(0, dur, n))).max())}
        for n in NS:
            res["s"][str(n)] = {"n": n, "max_util": float(U(np.linspace(0, 1, n)).max())}
        # time above the limit (diagnostic): fine uniform-s scan, time measure via ds / sd(s)
        sfine = np.linspace(0, 1, 2 ** 20 + 1)
        uf = U(sfine)
        _, _, _, xf, _ = T.evaluate(sfine)
        sdot = np.sqrt(np.maximum(xf, 0))
        mid_over = (uf[1:] > 1) & (uf[:-1] > 1)
        dtf = np.diff(sfine) * 2 / np.maximum(sdot[1:] + sdot[:-1], 1e-300)
        row = {"path": rec["path"], "grid": rec["grid"], "trim": tp.get("trim"), "K4": cr["K4"], "lb": cr["truth_lb"],
               "duration_s": dur, "time_above_limit_s": float(dtf[mid_over].sum()), "fine_s_max": float(uf.max()),
               "samplers": res}
        rows.append(row)
        print(json.dumps({"path": rec["path"][-22:], "grid": rec["grid"], "K4": cr["K4"], "lb-1": round(cr["truth_lb"] - 1, 6),
                          "t_above_ms": round(1e3 * row["time_above_limit_s"], 4),
                          **{k: round(v["max_util"] - 1, 5) for k, v in res["time"].items()},
                          "s32768": round(res["s"]["32768"]["max_util"] - 1, 5)}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "K4_HARD_LAYOUT_TOPPRA", "trim": tp.get("trim"), "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
