#!/usr/bin/env python3
"""Analyze K4 baselines: sampled checks vs the rigorous continuous-time answer (per population / regime).

Inputs: k4_baselines outputs (glob per population) and k4_toppra_certify outputs. Per population:
  truth classes   VIOLATED (rigorous lb > 1), CERTIFIED (verified ub <= 1), UNDECIDED (lb <= 1 < ub or no ub)
  sampled check S (exact model, N points, uniform-s or uniform-t; native float32 grids where available)
    miss          truth VIOLATED but S max <= 1                       (unsafe acceptance)
    gap           lb - S max  (how much of the true peak S does not see; utilization units)
    margin delta  S passes iff S max <= 1 - delta: misses and false rejections (truth CERTIFIED but rejected)
    N_needed      smallest N (uniform-s) that catches every rigorous violation in the population
  cost            compiled K4 producer + verifier seconds (1 thread) as recorded
Excess = lb - 1 (rigorous lower bound on how far the true peak exceeds the limit), reported for violations.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_formal_cohort import pct, proportion  # noqa: E402

DELTAS = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)


def classify(r):
    if r["truth_lb"] > 1:
        return "VIOLATED"
    if r.get("truth_ub") is not None and r["truth_ub"] <= 1 and r.get("truth_ub_verified", True) is not False:
        return "CERTIFIED"
    return "UNDECIDED"


def summarize(rows):
    n = len(rows)
    cls = [classify(r) for r in rows]
    viol = [r for r, c in zip(rows, cls) if c == "VIOLATED"]
    certd = [r for r, c in zip(rows, cls) if c == "CERTIFIED"]
    keys = sorted({k for r in rows for k in r["sampled"]}, key=lambda k: (k[0], int(k[1:])))
    checks = {}
    for k in keys:
        mx = np.array([r["sampled"][k]["max_util"] for r in rows])
        lb = np.array([r["truth_lb"] for r in rows])
        rec = {"gap": pct(list(lb - mx)), "by_margin": {}}
        for d in DELTAS:
            acc = mx <= 1 - d
            rec["by_margin"][str(d)] = {
                "misses": int(sum(a for a, c in zip(acc, cls) if c == "VIOLATED")),
                "false_rejections": int(sum((not a) for a, c in zip(acc, cls) if c == "CERTIFIED"))}
        checks[k] = rec
    nat = {}
    for g in ("native_grid_512", "S8193", "S32768"):
        vals = [(r, r["native_sampled"][g]["planning"]) for r in rows if r.get("native_sampled") and g in r["native_sampled"]]
        if vals:
            nat[g] = {"n": len(vals), "misses": sum(1 for r, v in vals if classify(r) == "VIOLATED" and v <= 1),
                      "false_alarms": sum(1 for r, v in vals if classify(r) == "CERTIFIED" and v > 1),
                      "gap": pct([r["truth_lb"] - v for r, v in vals])}
    sn = sorted(int(k[1:]) for k in keys if k[0] == "s")
    need = None
    for N in sn:
        if all(r["sampled"][f"s{N}"]["max_util"] > 1 for r in viol):
            need = N
            break
    return {"n": n, "truth": {c: cls.count(c) for c in ("VIOLATED", "CERTIFIED", "UNDECIDED")},
            "violation_rate": proportion(len(viol), n),
            "excess_over_limit_lb": pct([r["truth_lb"] - 1 for r in viol]) if viol else None,
            "K4_status": {s: sum(r["K4"] == s for r in rows) for s in sorted({r["K4"] for r in rows})},
            "bracket_width": pct([r["truth_ub"] - r["truth_lb"] for r in rows if r.get("truth_ub") is not None]),
            "k4_seconds_1thread": pct([r["k4_producer_seconds_1thread"] + r["k4_verifier_seconds_1thread"] for r in rows]),
            "sampled_checks": checks, "native_checks": nat, "uniform_s_N_needed_to_catch_all": need,
            "violations": [{"path": r["path"], "lb": r["truth_lb"], "ub": r.get("truth_ub"), "where": r["truth_lb_where"],
                            "s512": r["sampled"].get("s512", {}).get("max_util"),
                            "native512": (r.get("native_sampled") or {}).get("native_grid_512", {}).get("planning")}
                           for r in viol]}


def toppra_summary(rows):
    out = {}
    for g in sorted({r["grid"] for r in rows}, key=lambda x: (x != "auto", int(x) if x != "auto" else 0)):
        R = [r for r in rows if r["grid"] == g]
        ok = [r for r in R if r["K4"] != "TOPPRA_FAILED"]
        v = [r for r in ok if r["truth_lb"] > 1]
        out[g] = {"attempted": len(R), "toppra_failed": len(R) - len(ok), "n_grid": pct([r["n_grid"] for r in ok]),
                  "K4_status": {s: sum(r["K4"] == s for r in ok) for s in sorted({r["K4"] for r in ok})},
                  "violation_rate": proportion(len(v), len(ok)),
                  "excess_over_limit_lb": pct([r["truth_lb"] - 1 for r in v]) if v else None,
                  "gridpoint_max_util": pct([r["gridpoint_max_util"] for r in ok]),
                  "duration_s": pct([r["duration_s"] for r in ok if r["duration_s"] is not None]),
                  "duration_vs_curobo_timeopt_pct": pct([100 * (r["duration_s"] - r["native_duration_s"]) / r["native_duration_s"]
                                                        for r in ok if r["duration_s"] is not None]),
                  "toppra_seconds": pct([r["seconds"] for r in R])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", nargs="+", required=True, help="NAME=GLOB of k4_baselines outputs")
    ap.add_argument("--toppra", nargs="*", default=[], help="NAME=GLOB of k4_toppra_certify outputs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {"scope": "K4_BASELINES_ANALYSIS", "populations": {}, "toppra": {}}
    for spec in a.population:
        name, g = spec.split("=", 1)
        rows = [r for f in sorted(glob.glob(g)) for r in json.loads(Path(f).read_text())["rows"]]
        res["populations"][name] = summarize(rows)
        s = res["populations"][name]
        print(name, s["n"], s["truth"], "N_needed", s["uniform_s_N_needed_to_catch_all"],
              "excess", s["excess_over_limit_lb"] and {k: round(v, 6) for k, v in s["excess_over_limit_lb"].items() if k in ("p50", "max")})
    for spec in a.toppra:
        name, g = spec.split("=", 1)
        rows = [r for f in sorted(glob.glob(g)) for r in json.loads(Path(f).read_text())["rows"]]
        res["toppra"][name] = toppra_summary(rows)
        for gk, s in res["toppra"][name].items():
            print(name, gk, s["K4_status"], "viol", s["violation_rate"]["k"], "/", s["violation_rate"]["n"],
                  "excess p50/max", s["excess_over_limit_lb"] and (round(s["excess_over_limit_lb"]["p50"], 5), round(s["excess_over_limit_lb"]["max"], 5)),
                  "dur vs TimeOpt p50 %", s["duration_vs_curobo_timeopt_pct"]["p50"] and round(s["duration_vs_curobo_timeopt_pct"]["p50"], 2))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
