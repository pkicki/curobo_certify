# curobo_certify

Study continuous certification of unchanged cuRobo TimeOpt trajectories by adaptive Bernstein subdivision, with explicit measurement of added latency.

## Start here

- [Implementation and experiment plan](docs/CAMPAIGN_PLAN.md)
- [GPT-6 Luna handoff](docs/LUNA_PROMPT.md)
- [Progress and current evidence](docs/PROGRESS.md)
- [Interactive HTML report](reports/timeopt_certification_campaign.html) — download and open locally; GitHub displays HTML source.

The initial deliverable is a plan and a working educational report. No real GPU experiment has run. Analytical illustrations are not latency measurements or formal numerical certificates.

Build the offline report with `python tools/build_report.py`. Test its mathematics and aggregation with `node --test tests/*.test.cjs`. No external JavaScript or plotting service is needed. The compact measured-result import schema is in `schemas/report.schema.json`. The production certifier, GPU kernels and benchmark commands described in the plan are future work for Luna.

Upstream inspection pin: NVlabs/curobo `c301cb00a87012c2808b1a60649c4065c9b3c95a`. Use it as a dependency and preserve upstream license/attribution. The study starts with kinematic certificates; full torque/torque-rate certification needs rigorous dynamics enclosures.


## Independent recovery package (2026-11-10)

See [recovery gates](docs/RECOVERY_PLAN.md), [coding-agent handoff](docs/AGENT_RECOVERY_PROMPT.md) and the dependency-free `src/curobo_certify/boundary_reference.py` reference and `scripts/replay_boundary.py`. Run `python -m unittest discover -s tests -p 'test_*.py' -q` and `python -m compileall -q src scripts tests` for the new Python components (Node report tests separately). The `BOUNDARY_PASS` result is a limited boundary/regularity sub-contract; it is **not** native parity or a full K3 certificate. The unpublished local experiment at `ece5a3a` was unavailable on GitHub; do not mistake this separate recovery branch for its continuation. No native measured results are claimed.

Exact-rational K3 reference (analytic only): `src/curobo_certify/exact_k3_reference.py`, `scripts/replay_k3_reference.py` and `fixtures/boundary_multispan_analytic.json`. The script verifies completeness of exact Bernstein proof leaves for serialized cubic q(s) and x(s); **it does not verify native cuRobo FP32 parity or torque/torque-rate.** To try: `python scripts/replay_k3_reference.py fixtures/boundary_multispan_analytic.json`.


## Native jerk debugging and exploratory Bernstein tightness (2026-11-11)

The independent experimental branch adds a pinned-blob CUDA trace patch generator (scripts/make_timeopt_cuda_trace_patch.py) and bit-level parser (scripts/parse_timeopt_cuda_trace.py); **the native diagnostic CUDA extension has NOT been built/run here**. Apply the generated patch only to a separate disposable cuRobo checkout. The original upstream remains untouched.

It also adds mathematical-model tools (src/curobo_certify/bernstein_tightness.py, scripts/explore_k3_models.py), an exploratory exact-reference CPU hook benchmark (scripts/bench_frozen_model_cpu.py), and a standalone offline HTML visualization builder (scripts/render_tightness_html.py).

From repo root, run:
    python -m unittest discover -s tests -p 'test_*.py' -q
    python scripts/explore_k3_models.py fixtures/boundary_multispan_analytic.json --out /tmp/analytic_tightness.json
    python scripts/render_tightness_html.py /tmp/analytic_tightness.json --out /tmp/tightness.html
    python scripts/bench_frozen_model_cpu.py fixtures/boundary_multispan_analytic.json --warmups 1 --repeats 3 --out /tmp/analytic_cpu.json

**Every bundled example is analytic/synthetic.** The exact-rational maximum brackets apply to serialized polynomial models, not CUDA FP32 native execution. CPU timing excludes native planning, transfers and complete requests. See docs/NATIVE_PARITY_AND_TIGHTNESS_PLAN.md and docs/CODING_AGENT_JERK_CPU_EXPLORATION.md.

The user-reported local branch exp/timeopt-cert-integration-20261110 at 018fa07c was not published on GitHub. Do not treat this independent branch as its descendant.

## Compiled K3 initial-hull research package (2026-11-11)

This independent branch adds a C++ binary64 interval initial Bernstein-hull component, the exact-rational input/ctypes adapter, a stage profiler and an empirical native jerk operand attribution tool. It **cannot issue a complete K3 or native-linked certificate**. Use docs/COMPILED_HULL_AND_NATIVE_LINKAGE_PLAN_20261111.md and docs/GPT6_CODING_AGENT_COMPILED_CAMPAIGN.md.

Build with: python scripts/build_compiled_hull.py --out /tmp/libk3hull.so --manifest /tmp/build.json
Run tests with: python -m unittest discover -s tests -p 'test_*.py' -q
Profile a standalone analytic fixture with: python scripts/bench_compiled_hull.py fixtures/boundary_multispan_analytic.json --lib /tmp/libk3hull.so --out /tmp/hull.json
Render a standalone offline report with: python scripts/render_compiled_profile_html.py /tmp/hull.json --out /tmp/hull.html

The user-reported latest real Panda integration at 54e3905 is **unpublished** and must be preserved during cherry-pick. Only illustrative synthetic timing and mathematical-model differential unit tests were run in the independent environment.
