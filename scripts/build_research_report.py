#!/usr/bin/env python3
"""Build the standalone offline research report (reports/native_certification_research_report.html).

Embeds: exact BigInt proof engine (web/proof_core.js), exact model reconstruction
(web/model_core.js), the report app (web/research_app.js), and a JSON payload made
only from recorded result files (each listed with its SHA256) plus the frozen
float32 controls of every real path. Missing inputs are recorded as NOT_AVAILABLE,
never replaced with synthetic numbers.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.native_error_bound import exact_timing_table  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path, sources: dict):
    if not p.exists():
        sources[str(p.relative_to(ROOT))] = "NOT_AVAILABLE"
        return None
    sources[str(p.relative_to(ROOT))] = sha(p)
    return json.loads(p.read_text())


def path_payload(npz: Path, meta: dict | None, samples: int, planning_scale: float = 0.95) -> dict:
    with np.load(npz, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    geo = a["effective_geometry"].astype("<f4")
    lim = {n: np.minimum(np.abs(a[k][0]), np.abs(a[k][1])).astype(np.float32)
           for n, k in (("velocity", "velocity_limits"), ("acceleration", "acceleration_limits"), ("jerk", "jerk_limits"))}
    st = a["diagnostic_32768_states"]
    sp = ROOT / "runs/native_linkage/sample_progress" / f"{npz.stem}.npz"
    if sp.exists():  # true native internal s recovered by the bitwise-reproducing harness
        with np.load(sp) as zz:
            if int(zz["bit_mismatches"]) != 0:
                raise SystemExit(f"progress recovery not bitwise for {npz.stem}")
            prog = zz["s"].astype(np.float32)
        prog_label = "native internal progress s (recovered bitwise by harness)"
    else:
        prog = a["diagnostic_32768_progress"]
        prog_label = "kernel sample_progress (NOT internal s; first span clamped)"
    n = st.shape[1]
    idx = set(np.linspace(0, n - 1, samples).astype(int).tolist())
    for f in (1, 2, 3):  # always keep the sampled extremum of each family/joint
        idx.update(np.argmax(np.abs(st[f]), axis=0).tolist())
    idx = np.array(sorted(idx))
    sq = meta.get("source_query", {}) if meta else {}
    return {"id": npz.stem, "npz_sha256": sha(npz), "shape": list(geo.shape),
            "geometry_f32": base64.b64encode(geo.tobytes()).decode(),
            "timing_f32": base64.b64encode(a["timing_controls"].astype("<f4").tobytes()).decode(),
            "limits": {k: [str(Fraction(float(v))) for v in vs] for k, vs in lim.items()},
            "limits_float": {k: [float(v) for v in vs] for k, vs in lim.items()},
            "duration": float(a["native_duration"][0]),
            "scene": sq.get("scene_id") or (meta or {}).get("scene_config"), "magnitude": sq.get("magnitude_stratum", "small(pilot)"),
            "condition": sq.get("solver_condition", "default_margin_0p95"),
            "planning_scale": float(planning_scale), "pair_id": sq.get("pair_id"),
            "samples": {"progress": base64.b64encode(prog[idx].astype("<f4").tobytes()).decode(),
                        "vaj": base64.b64encode(st[1:4][:, idx].astype("<f4").tobytes()).decode(), "count": int(len(idx)),
                        "label": "SAMPLED_DIAGNOSTIC native S32768 (subsampled + per-joint extrema); x = " + prog_label}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "reports/native_certification_research_report.html")
    ap.add_argument("--samples", type=int, default=256)
    a = ap.parse_args()
    sources = {}
    R = ROOT / "runs"
    data = {
        "tightness": load(R / "hardcase_seed20261116/tightness_79paths.json", sources),
        "native_reported": load(R / "native_linkage/native_reported_16paths.json", sources),
        "native_reported_4ulp": load(R / "native_linkage/native_reported_16paths_4ulp_sensitivity.json", sources),
        "falsify": load(R / "native_linkage/same_s_falsify_16paths.json", sources),
        "time_consistency": load(R / "native_linkage/time_consistency_16paths.json", sources),
        "bench_summary": load(R / "benchmarks/summary.json", sources),
        "bench_native_all": load(R / "benchmarks/frozen_native_79paths.json", sources),
        "campaigns": {n: load(R / f"compiled_k3/differential_10k_{n}.json", sources) for n in ("O3", "O0", "native")},
        "browser": load(R / "browser_qa/proof_lab/browser_acceptance.json", sources),
        "fresh_3090": load(R / "benchmarks/fresh_rtx3090.json", sources),
        "fresh_a100": load(R / "benchmarks/fresh_a100.json", sources),
        "answers": load(ROOT / "reports/research_answers.json", sources),
        "margin_analysis": load(R / "margins/analysis.json", sources),
        "margin_tightness": load(R / "margins/tightness_145paths.json", sources),
        "margin_bench": load(R / "margins/cert_bench.json", sources),
        "time_audit": load(R / "margins/time_inversion_audit_224paths.json", sources),
        "l1_summary": load(R / "margins/l1_summary.json", sources),
        "fp_unit_audit": load(R / "margins/fp_unit_audit_rtx3090.json", sources),
        "sass_audit": load(R / "margins/sass_audit_rtx3090.json", sources),
        "margin_answers": load(ROOT / "reports/margin_campaign_answers.json", sources),
        "campaigns_mt4": {n: load(R / f"compiled_k3/differential_10k_{n}_mt4.json", sources) for n in ("O3", "O0", "native")},
    }
    if data["margin_bench"]:
        for r in data["margin_bench"]["rows"]:
            for k in list(r):
                if k.startswith("repeats_ms_") and k != "repeats_ms_total":
                    r[k] = [float(np.median(r[k]))]

    # strip bulky raw arrays
    if data["native_reported"]:
        for r in data["native_reported"]["rows"]:
            r.pop("per_span_joint_bounds", None)
    if data["bench_native_all"]:
        for r in data["bench_native_all"]["rows"]:
            for k in list(r):
                if k.startswith("repeats_ms_") and k != "repeats_ms_total":
                    r[k] = [float(np.median(r[k]))]
    if data["fresh_3090"]:
        for r in data["fresh_3090"]["rows"]:
            r.pop("repeats", None)
    if data["fresh_a100"]:
        for r in data["fresh_a100"]["rows"]:
            r.pop("repeats", None)
    t = exact_timing_table()
    table = [[[str(t[m][3 - p][k]) for k in range(4)] for p in range(4)] for m in range(29)]
    paths = []
    for npz in sorted((R / "native_pilot_resolved").glob("pilot-panda-0*.npz")) + \
               sorted((R / "native_panda_exploratory_seed20261111").glob("*.npz")) + \
               sorted((R / "hardcase_seed20261116/native").glob("*.npz")):
        meta_p = npz.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
        paths.append(path_payload(npz, meta, a.samples))
        sources[str(npz.relative_to(ROOT))] = paths[-1]["npz_sha256"]
    mq, md = R / "margins/queries.json", R / "margins/native"
    if mq.exists() and md.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        from bench_certification_methods import margin_index_from_records
        idx = margin_index_from_records(mq, md)
        for npz in sorted(md.glob("*.npz")):
            meta = json.loads(npz.with_suffix(".json").read_text())
            scale = idx[sha(npz)][0]
            paths.append(path_payload(npz, meta, a.samples, scale))
            sources[str(npz.relative_to(ROOT))] = paths[-1]["npz_sha256"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {"meta": {"branch": branch, "built_from_commit": head, "upstream": "NVlabs/curobo@c301cb00a87012c2808b1a60649c4065c9b3c95a",
                        "sources": sources}, "data": data, "table": table, "paths": paths}
    tpl = (ROOT / "web/research_template.html").read_text()
    parts = {"/* __PROOF_CORE__ */": (ROOT / "web/proof_core.js").read_text(),
             "/* __MODEL_CORE__ */": (ROOT / "web/model_core.js").read_text(),
             "/* __RESEARCH_APP__ */": (ROOT / "web/research_app.js").read_text(),
             "/* __DATA__ */": "window.REPORT_DATA=" + json.dumps(payload, separators=(",", ":")).replace("</", "<\\/") + ";"}
    for k, v in parts.items():
        if tpl.count(k) != 1:
            raise SystemExit(f"placeholder {k} missing/duplicated")
        if "</script" in v.lower() and k != "/* __DATA__ */":
            raise SystemExit("script terminator in source")
        tpl = tpl.replace(k, v)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(tpl, encoding="utf-8")
    print(f"Built {a.out} ({a.out.stat().st_size/1e6:.2f} MB), {len(paths)} paths, sources {len(sources)}")


if __name__ == "__main__":
    main()
