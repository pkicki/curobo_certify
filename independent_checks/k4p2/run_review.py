"""Run the P2 falsification campaign and write evidence JSON.

  PYTHONPATH=<src of the reviewed snapshot>:. python independent_checks/k4p2/run_review.py --out X.json
The memo's numbers were produced against a read-only `git archive 6117b6b` snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT))
from independent_checks.k4p2 import campaign as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contract", type=Path, default=ROOT / "docs/reports/k4/p0/panda_model_contract_v2.json")
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--boxes", type=int, default=40)
    ap.add_argument("--box-samples", type=int, default=12)
    ap.add_argument("--path-samples", type=int, default=80)
    ap.add_argument("--synthetic", type=int, default=30)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("refusing to overwrite")
    import curobo_certify
    from curobo_certify.k4_panda_model import PandaModel, load_contract
    from curobo_certify.k4_path import certify_path
    from curobo_certify.k4_verify import WorldFrameDynamics, verify_k4, verify_k4_witness
    src = Path(curobo_certify.__file__).parent
    reviewed = {p: hashlib.sha256((src / p).read_bytes()).hexdigest() for p in
                ("k4_interval.py", "k4_panda_model.py", "k4_path.py", "k4_verify.py", "k4_reference.py",
                 "spline_polynomials.py", "boundary_reference.py")}
    c = load_contract(a.contract)
    M, W = PandaModel(c), WorldFrameDynamics(c)
    orc, orm = C.oracle("iv", c), C.oracle("mp", c)
    rng = np.random.default_rng(20261117)
    prng = random.Random(20261117)
    out = {"scope": "SAMPLED FALSIFICATION against a rigorous (iv) / 160-bit (mp) Lagrangian oracle; not a proof",
           "src_dir": str(src), "reviewed_sha256": reviewed, "contract_sha256": c["contract_sha256"]}
    t0 = time.time()
    st, bad = C.check_states(orc, M, W, rng, a.states)
    out["exact_states"] = {"n": a.states, "oracle": "iv", "verdicts": st, "bad": bad}
    print("states", st, len(bad), flush=True)
    st, bad = C.check_boxes(orm, M, W, rng, a.boxes, a.box_samples)
    out["sub_boxes"] = {"n_boxes": a.boxes, "samples_per_box": a.box_samples, "oracle": "mp160", "verdicts": st, "bad": bad}
    print("boxes", st, len(bad), flush=True)
    spans = C.fixture_spans()
    lo = np.array([-87.0] * 4 + [-12.0] * 3)
    hi = -lo
    paths = []
    r = certify_path(spans, M, lo, hi)
    st, bad = C.check_path_result(orc, spans, r, lo, hi, prng, a.path_samples)
    v = verify_k4(spans, c, r, lo, hi)
    paths.append({"case": "fixture_physical", "status": r.status, "nodes": r.nodes, "worst": r.worst_utilization,
                  "verify": v["pass"], "verdicts": st, "bad": bad})
    peak = max(r.worst_utilization)
    for f in (0.3, 0.5, 0.6, peak * 0.999, peak * 0.99, peak * 1.001, peak * 1.01, 0.7, 0.95):
        L, H = lo * f, hi * f
        rr = certify_path(spans, M, L, H, max_depth=10)
        st, bad = C.check_path_result(orc, spans, rr, L, H, prng, 12)
        vv = (verify_k4(spans, c, rr, L, H, expect_physical=False)["pass"] if rr.status == "CERTIFIED"
              else verify_k4_witness(spans, c, rr, L, H)["pass"] if rr.status == "VIOLATED_TORQUE" else None)
        paths.append({"case": f"fixture_scaled_{f:.6f}", "status": rr.status, "nodes": rr.nodes,
                      "independent_verifier": vv, "verdicts": st, "bad": bad})
        print(paths[-1]["case"], rr.status, vv, st, len(bad), flush=True)
    for k in range(a.synthetic):
        kind = ["pos", "pos", "zero_start", "neg_pocket", "zero"][k % 5]
        sp = C.synthetic_spans(prng, prng.choice([1, 2, 3]), kind)
        f = prng.choice([0.2, 0.5, 1.0])
        L, H = lo * f, hi * f
        rr = certify_path(sp, M, L, H, max_depth=8, max_nodes=20000)
        st, bad = C.check_path_result(orc, sp, rr, L, H, prng, 6)
        vv = (verify_k4(sp, c, rr, L, H, expect_physical=False)["pass"] if rr.status == "CERTIFIED"
              else verify_k4_witness(sp, c, rr, L, H)["pass"] if rr.status == "VIOLATED_TORQUE" else None)
        paths.append({"case": f"synthetic_{k}_{kind}_f{f}", "status": rr.status, "nodes": rr.nodes,
                      "independent_verifier": vv, "verdicts": st, "bad": bad})
        print(paths[-1]["case"], rr.status, vv, st, len(bad), flush=True)
    out["paths"] = paths
    out["wall_s"] = round(time.time() - t0, 1)
    tot_bad = len(out["exact_states"]["bad"]) + len(out["sub_boxes"]["bad"]) + sum(len(p["bad"]) for p in paths)
    out["total_counterexamples_or_ambiguous"] = tot_bad
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("TOTAL bad", tot_bad, "wall", out["wall_s"], flush=True)


if __name__ == "__main__":
    main()
