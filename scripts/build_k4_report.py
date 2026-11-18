#!/usr/bin/env python3
"""Build the offline interactive K4 torque-certification report (single self-contained HTML).

Real evidence only: P0 contract/parity, P3 comparison summary, P4 population analysis, P6 retiming,
P5 fresh latency, review memos. Per primary output it recomputes (CPU, deterministic): dense rigorous point
torques (24/span; SAMPLED points of the exact model, enclosure midpoints), the certified tm1 leaf torque
enclosures per joint (RIGOROUS band), and x(s) for native and accepted certified-retimed controls.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from curobo_certify.k4_panda_model import PandaModel, load_contract  # noqa: E402
from curobo_certify.k4_path import _point_boxes, torque_enclosure  # noqa: E402
from curobo_certify.k4_reference import dpoly, eval_poly  # noqa: E402
from curobo_certify.k4_reuse import GeometryCache, _leaf_bounds  # noqa: E402
from curobo_certify.spline_polynomials import extract_timeopt_spans  # noqa: E402


def r5(x):
    return float(f"{x:.5g}")


def path_payload(M, npz, lo, hi, retimed_hex=None, per_span=12):
    with np.load(npz) as z:
        g, tc = z["effective_geometry"], z["timing_controls"].astype(np.float32)
    spans = extract_timeopt_spans(g, tc)
    pts, rl, ss = [], [], []
    for sp in spans:
        h = sp.right - sp.left
        raw = ([tuple(p) for p in sp.q] + [dpoly(p, 1, h) for p in sp.q] + [dpoly(p, 2, h) for p in sp.q]
               + [tuple(sp.x), dpoly(sp.x, 1, h)])
        for t in range(per_span):
            u = Q(t, per_span - 1)
            pts.append(u); rl.append(raw); ss.append(float(sp.left + h * u))
    pq, pqp, pqpp, px, pxp, ex = _point_boxes(rl, pts, 7)
    tau = torque_enclosure(M, pq, pqp, pqpp, px, pxp)
    T = np.stack([0.5 * (t.lo + t.hi) for t in tau], 1)
    C = GeometryCache(M, spans, "tm1")
    r = C.certify(spans, lo, hi)
    band = []
    for (si, path, blo, bhi) in r.leaves:
        a, b = _leaf_bounds(path)
        s0 = float(spans[si].left + (spans[si].right - spans[si].left) * a)
        s1 = float(spans[si].left + (spans[si].right - spans[si].left) * b)
        band.append([r5(s0), r5(s1), [r5(v) for v in blo], [r5(v) for v in bhi], len(path)])
    x = [r5(float(e[21])) for e in ex]
    xr = None
    if retimed_hex:
        tr = np.array([float.fromhex(h) for h in retimed_hex], dtype=np.float32)
        sr = extract_timeopt_spans(g, tr)
        xr = []
        for sp in sr:
            for t in range(per_span):
                xr.append(r5(float(eval_poly(sp.x, Q(t, per_span - 1)))))
    return {"s": [r5(v) for v in ss], "tau": [[r5(v) for v in row] for row in T.T.tolist()], "band": band,
            "x": x, "x_retimed": xr, "status": r.status, "nodes": r.nodes, "max_depth": r.max_depth,
            "bound_util": [r5(v) for v in r.worst_utilization]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--native-dir", type=Path, required=True)
    ap.add_argument("--retime", type=Path, nargs="*", default=[])
    ap.add_argument("--ext", type=Path, help="k4_analyze_extensions output (identified model, synthetic payloads)")
    ap.add_argument("--payload-native-dir", type=Path, help="payload NPZ directory: adds payload primary outputs to the explorer")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/k4_torque_certification_report.html")
    a = ap.parse_args()
    c = load_contract(ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
    M = PandaModel(c)
    lo = np.array([float(Q(v)) for v in c["model"]["effort_lower"]])
    hi = np.array([float(Q(v)) for v in c["model"]["effort_upper"]])
    an = json.loads(a.analysis.read_text())
    ret = {}
    for p in a.retime:
        d = json.loads(p.read_text())
        for r in d["rows"]:
            ret.setdefault(r["npz_sha256"], {})[d.get("limit_scale", "1.00")] = r
    paths = []
    for arm in ("primary", "secondary", "stress"):
        for row in an["arms"][arm]["rows"]:
            npz = a.native_dir / f"{row['source_id']}.npz"
            rr = ret.get(row["npz_sha256"], {})
            hexes = (rr.get("1.00") or {}).get("accepted_timing_float32_hex")
            pp = path_payload(M, npz, lo, hi, hexes)
            paths.append({**{k: row[k] for k in ("source_id", "pair_id", "scene", "magnitude", "npz_sha256", "duration",
                                                 "MODEL_K3", "MODEL_K4", "K3_PLUS_K4", "PLANNING_K4", "k4_bound_util",
                                                 "rigorous_point_max_util", "native_sampled_max_util_physical",
                                                 "native_sampled_by_grid", "k4_witness")},
                          "arm": arm, "retime": {k: {kk: v.get(kk) for kk in ("accepted_lambda", "gain_vs_native_pct",
                                                                             "retimed_duration_bracket_s")}
                                                 for k, v in rr.items()}, **pp})
            print(row["source_id"], pp["status"], flush=True)
    plo = [float.fromhex(h) for h in c["model"]["timeopt_torque_lower_scaled_float32_hex"]]
    phi = [float.fromhex(h) for h in c["model"]["timeopt_torque_upper_scaled_float32_hex"]]
    for pth in paths:
        pth["lim"] = {"lo": lo.tolist(), "hi": hi.tolist(), "plo": plo, "phi": phi}
        pth["model"] = "cuRobo contract v2"
    ex = json.loads(a.ext.read_text()) if a.ext else None
    if ex and a.payload_native_dir:
        from curobo_certify.k4_payload import with_payload
        base_c = json.loads((ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json").read_text())
        for v, d in ex["variants"].items():
            if not (v.startswith("cu_") and d["contract"].get("payload")):
                continue
            pc = with_payload(base_c, d["contract"]["payload"])
            if pc["contract_sha256"] != d["contract"]["contract_sha256"]:
                raise SystemExit("payload contract hash mismatch")
            PM = PandaModel(pc)
            for row in d["arms"]["primary"]["rows"]:
                pp = path_payload(PM, a.payload_native_dir / f"{row['source_id']}.npz", lo, hi, None)
                paths.append({**{k: row.get(k) for k in ("source_id", "pair_id", "npz_sha256", "duration", "MODEL_K3", "MODEL_K4",
                                                       "K3_PLUS_K4", "PLANNING_K4", "k4_bound_util", "rigorous_point_max_util",
                                                       "native_sampled_max_util_physical")},
                              "scene": "payload", "magnitude": d["contract"]["payload"], "k4_witness": row.get("planning_witness") or row.get("witness"),
                              "native_sampled_by_grid": {"native_grid_512": row.get("native_grid512_max_util_physical")},
                              "arm": "primary", "variant": v, "retime": {}, "model": pc["model"]["scope"],
                              "lim": {"lo": lo.tolist(), "hi": hi.tolist(), "plo": plo, "phi": phi}, **pp})
                print(v, row["source_id"], pp["status"], flush=True)
    data = {
        "analysis": {k: v for k, v in an.items() if k != "arms"} | {"arms": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                                                           for k, v in an["arms"].items()}},
        "paths": paths,
        "ext": ({**ex, "variants": {n: {**v, "arms": {k: {kk: vv for kk, vv in x.items() if kk != "rows"} for k, x in v["arms"].items()}}
                                     for n, v in ex["variants"].items()}} if ex else None),
        "contract": {"sha256": c["contract_sha256"], "findings": c["physical_validity_findings"],
                     "status": c["physical_validity_status"], "provenance": c["provenance"],
                     "crosscheck": c["float32_crosscheck_vs_curobo"],
                     "links": [{k: e[k] for k in ("name", "mass", "com", "inertia", "inertial_source", "origin_xyz", "origin_rpy")}
                               for e in c["model"]["links"]]},
        "parity": json.loads((ROOT / "docs/reports/k4/p0/parity.json").read_text()),
        "p3": {k: v for k, v in json.loads((ROOT / "docs/reports/k4/p3/summary.json").read_text()).items() if k != "per_row"},
        "limits": {"lower": lo.tolist(), "upper": hi.tolist()},
    }
    tpl = (ROOT / "web/k4_report_template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), allow_nan=False, default=float))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html)
    import hashlib
    print(a.out, f"{len(html) / 1e6:.2f} MB", hashlib.sha256(html.encode()).hexdigest())


if __name__ == "__main__":
    main()
