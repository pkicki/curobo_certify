#!/usr/bin/env python3
"""Exact per-residual tightness analysis of frozen native TimeOpt outputs.

For each path and contract (physical, planning=0.95x physical) and every K3
residual (span x family x joint) of the exact serialized model:
  U0          exact initial Bernstein hull upper bound (depth 0)
  [L*, U*]    exact bracket of the true residual maximum (best-first exact
              de Casteljau; L* from exact point values, U* from leaf hulls)
  class       H0_CERTIFIED (U0<=0) | SPLIT_CERTIFIABLE (U*<=0<U0) |
              GENUINE_VIOLATION (L*>0, exact witness) | UNRESOLVED (L*<=0<U*)
Native dense S32768 sampled utilization is reported as a SAMPLED_DIAGNOSTIC.
Also writes a Proof Lab bundle of genuine real residuals that need subdivision.
Scope: EXACT_SERIALIZED_MODEL (not native-linked by itself).
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import multiprocessing as mp
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.exact_k3_reference import bernstein_coeffs, build_residuals, midpoint_split  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


def peak_bracket(b, budget=4000, tol=Fraction(1, 10 ** 9)):
    """Exact [L*, U*] for max of a Bernstein polynomial on [0,1]; depth info."""
    lower = max(b[0], b[-1])
    heap = [(-max(b), 0, tuple(b), 0)]
    counter, steps, deepest = 1, 0, 0
    while heap and steps < budget:
        neg_up, _, c, depth = heapq.heappop(heap)
        up = -neg_up
        if up - lower <= tol or up <= lower:
            heapq.heappush(heap, (neg_up, counter, c, depth))
            break
        L, R = midpoint_split(c)
        lower = max(lower, L[-1], c[0], c[-1])
        for child in (L, R):
            heapq.heappush(heap, (-max(child), counter, child, depth + 1)); counter += 1
        deepest = max(deepest, depth + 1)
        steps += 1
    upper = max(-h[0] for h in heap) if heap else lower
    return lower, max(upper, lower), deepest, steps


def classify(U0, L, U, strict):
    if strict:  # -x <= 0 with strict interior negativity
        if U0 < 0 or (U0 <= 0 and False):
            return "H0_CERTIFIED"
    if U0 <= 0:
        return "H0_CERTIFIED"
    if L > 0:
        return "GENUINE_VIOLATION"
    if U <= 0:
        return "SPLIT_CERTIFIABLE"
    return "UNRESOLVED"


_SCALES: dict = {}


def analyze(path_str: str) -> dict:
    path = Path(path_str)
    raw = path.read_bytes()
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    spans = extract_timeopt_spans(a["effective_geometry"], a["timing_controls"])
    phys = {n: np.minimum(np.abs(a[k][0]), np.abs(a[k][1])).astype(np.float32)
            for n, k in (("velocity", "velocity_limits"), ("acceleration", "acceleration_limits"), ("jerk", "jerk_limits"))}
    sha = hashlib.sha256(raw).hexdigest()
    # Planning contract uses the source-hash-backed TimeOpt scale of THIS output (0.95 pinned default otherwise).
    scale = np.float32(_SCALES.get(sha, (0.95,))[0])
    contracts = {"physical": phys, "planning": {k: v * scale for k, v in phys.items()}}
    states = a["diagnostic_32768_states"].astype(np.float64)
    out = {"path": path.stem, "source_npz_sha256": sha, "planning_scale": float(scale),
           "source_query_id": _SCALES.get(sha, (None, None))[1],
           "native_duration_s": float(a["native_duration"][0]), "common_spans": len(spans), "contracts": {}}
    bundle = []
    for cname, lim in contracts.items():
        sampled = {fam: float(np.max(np.abs(states[i]) / lim[fam].astype(np.float64)[None, :]))
                   for i, fam in ((1, "velocity"), (2, "acceleration"), (3, "jerk"))}
        rows, counts = [], {}
        worst = {}
        for si, sp in enumerate(spans):
            for fam, joint, power in build_residuals(sp, [float(v) for v in lim["velocity"]],
                                                     [float(v) for v in lim["acceleration"]], [float(v) for v in lim["jerk"]]):
                b = bernstein_coeffs(power)
                U0 = max(b)
                key = fam if fam != "positive_x" else "positive_x"
                if U0 <= 0:
                    cls, L, U, depth = "H0_CERTIFIED", None, None, 0
                    worst.setdefault(key, U0)
                    worst[key] = max(worst[key], U0)
                else:
                    L, U, depth, _ = peak_bracket(b)
                    cls = classify(U0, L, U, fam == "positive_x")
                    worst[key] = max(worst.get(key, U0), U0)
                    rows.append({"span": si, "family": fam, "joint": joint, "U0": float(U0),
                                 "peak_lower": float(L), "peak_upper": float(U), "class": cls,
                                 "hull_gap_bracket": [float(U0 - U), float(U0 - L)], "bracket_depth": depth,
                                 "s_interval": [float(sp.left), float(sp.right)]})
                    if len(bundle) < 250:
                        bundle.append({"id": f"{path.stem}-{cname}-s{si}-{fam}-j{joint}",
                                       "title": f"{path.stem} · {cname} · span {si} · {fam} j{joint}",
                                       "family": fam, "strict": fam == "positive_x",
                                       "strict_left": fam == "positive_x" and si > 0,
                                       "strict_right": fam == "positive_x" and si < len(spans) - 1,
                                       "coefficients": [str(c) for c in b],
                                       "description": (f"REAL native TimeOpt output {path.stem} (NPZ sha256 "
                                                       f"{out['source_npz_sha256'][:16]}...), exact serialized model, "
                                                       f"{cname} limits, span [{sp.left}, {sp.right}]. Initial hull "
                                                       f"inconclusive (U0={float(U0):.3g}); class {cls}. Single residual "
                                                       "only; NOT a whole-path or native certificate."),
                                       "scope": "REAL_NATIVE_OUTPUT_EXACT_MODEL_SINGLE_RESIDUAL"})
                counts[cls] = counts.get(cls, 0) + 1
        out["contracts"][cname] = {"class_counts": counts, "nontrivial_residuals": rows,
                                   "max_initial_hull_upper": {k: float(v) for k, v in worst.items()},
                                   "sampled_max_utilization_S32768": sampled}
    out["bundle"] = bundle
    return out


def _set_scales(s):
    global _SCALES
    _SCALES = s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle-out", type=Path)
    ap.add_argument("--margin-queries", type=Path)
    ap.add_argument("--margin-records-dir", type=Path)
    a = ap.parse_args()
    global _SCALES
    if a.margin_queries:
        sys.path.insert(0, str(ROOT / "scripts"))
        from bench_certification_methods import margin_index_from_records
        _SCALES = {k: (v[0], v[1]) for k, v in margin_index_from_records(a.margin_queries, a.margin_records_dir).items()}
    with mp.Pool(a.workers, initializer=_set_scales, initargs=(_SCALES,)) as pool:
        rows = pool.map(analyze, [str(p) for p in a.npz])
    bundle = []
    for r in rows:
        bundle.extend(r.pop("bundle"))
    summary = {"scope": "EXACT_SERIALIZED_MODEL_TIGHTNESS_EXPLORATORY", "paths": len(rows), "rows": rows}
    for c in ("physical", "planning"):
        tot = {}
        for r in rows:
            for k, v in r["contracts"][c]["class_counts"].items():
                tot[k] = tot.get(k, 0) + v
        summary[f"{c}_class_totals"] = tot
        summary[f"{c}_paths_needing_subdivision"] = sum(
            any(x["class"] != "H0_CERTIFIED" for x in r["contracts"][c]["nontrivial_residuals"]) for r in rows)
        summary[f"{c}_paths_with_genuine_violation"] = sum(
            any(x["class"] == "GENUINE_VIOLATION" for x in r["contracts"][c]["nontrivial_residuals"]) for r in rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1) + "\n")
    if a.bundle_out:
        bundle.sort(key=lambda e: (e["description"].count("GENUINE_VIOLATION") == 0, e["id"]))
        a.bundle_out.write_text(json.dumps({"schema_version": 1, "scope": "REAL_NATIVE_OUTPUT_EXACT_MODEL_SINGLE_RESIDUAL_DEMOS",
                                            "native_parity": "SEE_NATIVE_REPORTED_CERTIFICATE_SEPARATELY",
                                            "examples": bundle[:250]}, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
