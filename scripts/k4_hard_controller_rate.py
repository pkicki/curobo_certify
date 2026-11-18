#!/usr/bin/env python3
"""K4 HARD CASES (3): controller-rate sampled checks (250 Hz, 500 Hz, 1 kHz, 4 kHz uniform-time) on cuRobo outputs.

For every output with a k4_baselines record (rigorous bracket), exact-model torque (float64) is evaluated at the
uniform-time samples a real-time controller would use (cuRobo's own progress table maps time to path parameter).
A violation (rigorous lb > 1) is MISSED at rate r when all controller-rate samples are within the limit. Also reports the
fraction of the true excess the sampler sees."""
import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RATES = (250, 500, 1000, 4000)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", nargs="+", required=True, help="NAME=BASELINE_GLOB=NPZ_DIR")
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from curobo_certify import k4_compiled as KC
    from curobo_certify import k4_sampling as KS
    from curobo_certify.k4_panda_model import PandaModel
    from curobo_certify.spline_polynomials import extract_timeopt_spans
    c = json.loads(a.contract.read_text())
    if a.payload:
        from curobo_certify.k4_payload import with_payload
        c = with_payload(c, a.payload)
    M = PandaModel(c)
    lo = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]])
    hi = np.array([float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]])
    tmp = tempfile.TemporaryDirectory(); KC.build(Path(tmp.name) / "p.so")
    CC = KC.CompiledCoefficients(M, Path(tmp.name) / "p.so")
    out = {}
    for spec in a.population:
        name, g, nd = spec.split("=")
        rows = [r for f in sorted(glob.glob(g)) for r in json.loads(Path(f).read_text())["rows"]]
        res = []
        for r in rows:
            with np.load(Path(nd) / f"{r['path']}.npz") as z:
                T = KS.SpanTable(extract_timeopt_spans(z["effective_geometry"], z["timing_controls"]))
                times, prog = z["diagnostic_32768_times"], z["diagnostic_32768_progress"]
            dur = float(times[-1])
            rec = {"path": r["path"], "lb": r["truth_lb"], "ub": r.get("truth_ub"), "duration": dur, "rates": {}}
            for hz in RATES:
                n = int(np.floor(dur * hz)) + 1
                s = KS.grid_uniform_t(n, times, prog)
                rec["rates"][str(hz)] = float(KS.utilization(KS.sampled_torque(CC, T, s), lo, hi).max())
            res.append(rec)
        viol = [x for x in res if x["lb"] > 1]
        out[name] = {"n": len(res), "violated": len(viol),
                     "missed_at": {hz: sum(1 for x in viol if x["rates"][str(hz)] <= 1) for hz in map(str, RATES)},
                     "missed_meaningful_1e-4": {hz: sum(1 for x in viol if x["lb"] - 1 >= 1e-4 and x["rates"][str(hz)] <= 1) for hz in map(str, RATES)},
                     "max_missed_excess": {hz: max([x["lb"] - 1 for x in viol if x["rates"][str(hz)] <= 1] or [0]) for hz in map(str, RATES)},
                     "rows": res}
        print(name, {k: v for k, v in out[name].items() if k != "rows"}, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"scope": "K4_HARD_CONTROLLER_RATE", "rates_hz": RATES, "populations": out}, indent=1) + "\n")


if __name__ == "__main__":
    main()
