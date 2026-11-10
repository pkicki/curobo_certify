# curobo_certify

Research campaign for continuous certification of unchanged cuRobo TimeOpt trajectories by adaptive Bernstein subdivision, with explicit measurement of added latency.

## Current state

This branch contains the staged campaign plan, an offline educational report, outward-rounded CPU interval/Bernstein reference arithmetic, a fail-closed one-span K3 residual prototype, a leaf replay checker, and hash-verified corpus-ingestion safeguards. No native TimeOpt path has been certified and no GPU benchmark has run. The prototype reports `UNKNOWN_BOUNDARY` when its interior polynomial obligations pass; torque and torque rate remain `UNSUPPORTED`. Demo values are analytic illustrations, never benchmark data.

P0 adapter parity is blocked: Python 3.12.4 has no PyTorch/cuRobo installed, and the nearby cuRobo checkout is not the requested `c301cb00a87012c2808b1a60649c4065c9b3c95a` revision. GPU devices are visible but cannot be used for campaign evidence before the adapter/correctness gates pass. There is no frozen source cohort.

## Start here

- [Implementation and experiment plan](docs/CAMPAIGN_PLAN.md)
- [Progress and current evidence](docs/PROGRESS.md)
- [CPU interval assumptions](docs/CPU_INTERVAL_ASSUMPTIONS.md)
- [Offline interactive report](reports/timeopt_certification_campaign.html)

Build the report with `python tools/build_report.py`. CPU reference tests run with `python -m unittest discover -s tests -p 'test_*.py' -v`. The report's Node tests are in `tests/*.test.cjs`, but Node is absent in the captured environment. `python scripts/capture_environment.py --out runs/environment.json` records available environment facts. `scripts/freeze_corpus.py` only accepts real source rows and hashed arrays; it never creates substitute inputs. The benchmark entry point refuses to run before adapter parity and complete proof obligations pass.

Use the pinned upstream as a dependency and preserve NVIDIA attribution/license; do not copy or modify its implementation. Full torque/torque-rate certification needs separate rigorous dynamics enclosures.
