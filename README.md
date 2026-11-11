# curobo_certify

Research campaign for continuous certification of unchanged cuRobo TimeOpt trajectories by adaptive Bernstein subdivision, with explicit measurement of added latency.

## Current state

This branch contains the staged campaign plan, offline report, interval/Bernstein arithmetic scaffold, exact-rational boundary/K3 oracles, pinned native adapter, and a real source-query pilot bridge. Eight Panda native planner/TimeOpt outputs were generated; all fail the adapter's physical-jerk parity gate. Eight UR5e requests remain failed because the pin has no verified UR5/UR5e model asset. Serialized-model K3 proofs pass, but no native-linked certificate is counted. Torque and torque rate remain `UNSUPPORTED`; the subdivision lab is analytic and pilot plots are sampled diagnostics.

A new isolated `curobo-timeopt-c301cb` environment uses Python 3.11.17, PyTorch 2.5.1/CUDA 12.4, and the separate clean cuRobo checkout at `c301cb00a87012c2808b1a60649c4065c9b3c95a`. Native TimeOpt tests and the synthetic adapter smoke passed. The first real pilot is retained with explicit start/goal inputs and hashes. See [native query bridge results](docs/NATIVE_QUERY_BRIDGE.md). Dev40/Holdout100/Confirm300, certified-path latency and campaign acceptance remain NOT RUN.

## Start here

- [Implementation and experiment plan](docs/CAMPAIGN_PLAN.md)
- [Progress and current evidence](docs/PROGRESS.md)
- [CPU interval assumptions](docs/CPU_INTERVAL_ASSUMPTIONS.md)
- [Native planner-to-TimeOpt bridge and pilot](docs/NATIVE_QUERY_BRIDGE.md)
- [Offline interactive report](reports/timeopt_certification_campaign.html)

Build the pilot report with `PYTHONPATH=src python scripts/build_real_pilot_report_data.py --pilot-dir runs/native_pilot_resolved --out runs/native_pilot_resolved/report_pilot.json && python tools/build_report.py --pilot runs/native_pilot_resolved/report_pilot.json`. CPU tests run with `python -m unittest discover -s tests -p 'test_*.py' -v`. Run native source-query generation using the commands in `docs/NATIVE_QUERY_BRIDGE.md`. The synthetic native adapter smoke remains a separate gate. Capture environment/provenance with `scripts/capture_environment.py`. `scripts/freeze_corpus.py` only accepts real source rows and hashed arrays; it never creates substitute inputs. The benchmark entry point refuses to run before adapter parity and complete proof obligations pass.

Use the pinned upstream as a dependency and preserve NVIDIA attribution/license; do not copy or modify its implementation. The benchmark stub fails closed until a compatible Panda/UR5 TimeOpt cohort and complete measurement driver exist. Full torque/torque-rate certification needs separate rigorous dynamics enclosures.

## Independent recovery package

See [recovery gates](docs/RECOVERY_PLAN.md), [coding-agent handoff](docs/AGENT_RECOVERY_PROMPT.md), `boundary_reference.py` and `exact_k3_reference.py`. These dependency-free exact-rational implementations are analytic reference tools and do not establish native parity or campaign measurements. `BOUNDARY_PASS` alone is not a full K3 certificate.

## Native jerk debugging and exploratory Bernstein tightness (2026-11-11)

This integration includes a pinned-blob CUDA trace patch generator (`scripts/make_timeopt_cuda_trace_patch.py`) and parser (`scripts/parse_timeopt_cuda_trace.py`). The CUDA diagnostic extension has not yet been built or run; apply its patch only in a disposable cuRobo checkout. It also includes exact-rational Bernstein tightness analysis, a frozen-model CPU timing harness and an offline HTML visualization builder. Bundled fixtures are analytic. CPU timing is not native-linked complete-request latency. See `docs/NATIVE_PARITY_AND_TIGHTNESS_PLAN.md` and `docs/CODING_AGENT_JERK_CPU_EXPLORATION.md`.

## Compiled K3 initial-hull research package (2026-11-11)

This integration adds a C++ binary64 interval initial Bernstein-hull component, the exact-rational input/ctypes adapter, a stage profiler and an empirical native jerk operand attribution tool. It **cannot issue a complete K3 or native-linked certificate**. Use `docs/COMPILED_HULL_AND_NATIVE_LINKAGE_PLAN_20261111.md` and `docs/GPT6_CODING_AGENT_COMPILED_CAMPAIGN.md`.

Build with: `python scripts/build_compiled_hull.py --out /tmp/libk3hull.so --manifest /tmp/build.json`. Run `python -m unittest discover -s tests -p 'test_*.py' -q` for the Python suite. The standalone analytic profiler is `python scripts/bench_compiled_hull.py fixtures/boundary_multispan_analytic.json --lib /tmp/libk3hull.so --out /tmp/hull.json`.

All compiled hull results are partial model checks; they exclude full boundary/regularity proof replay and native linkage. Refer to `docs/PROGRESS.md` for the real Panda results and current blockers.
