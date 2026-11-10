# curobo_certify

Research campaign for continuous certification of unchanged cuRobo TimeOpt trajectories by adaptive Bernstein subdivision, with explicit measurement of added latency.

## Current state

This branch contains the staged campaign plan, offline report prototype, interval/Bernstein arithmetic scaffold, exact-rational boundary/K3 oracles, pinned native adapter, and hash-verified corpus-ingestion safeguards. One synthetic one-joint native smoke path passed adapter parity. It is not a Panda/UR5 cohort path or campaign measurement. Torque and torque rate remain `UNSUPPORTED`; demo values remain analytic illustrations.

A new isolated `curobo-timeopt-c301cb` environment uses Python 3.11.17, PyTorch 2.5.1/CUDA 12.4, and the separate clean cuRobo checkout at `c301cb00a87012c2808b1a60649c4065c9b3c95a`. Native TimeOpt tests and the RTX 3090 adapter parity smoke passed. Full campaign work remains blocked by the lack of frozen Panda/UR5 source-query cohorts; no Dev40/holdout latency result exists.

## Start here

- [Implementation and experiment plan](docs/CAMPAIGN_PLAN.md)
- [Progress and current evidence](docs/PROGRESS.md)
- [CPU interval assumptions](docs/CPU_INTERVAL_ASSUMPTIONS.md)
- [Offline interactive report](reports/timeopt_certification_campaign.html)

Build the report with `python tools/build_report.py`. CPU tests run with `python -m unittest discover -s tests -p 'test_*.py' -v`. Run the native adapter smoke using `PYTHONPATH=src conda run -n curobo-timeopt-c301cb python scripts/native_timeopt_parity.py`; it writes parity and frozen-array artifacts under `runs/`. Capture environment/provenance with `scripts/capture_environment.py`. `scripts/freeze_corpus.py` only accepts real source rows and hashed arrays; it never creates substitute inputs. The benchmark entry point refuses to run before adapter parity and complete proof obligations pass.

Use the pinned upstream as a dependency and preserve NVIDIA attribution/license; do not copy or modify its implementation. The benchmark stub fails closed until a compatible Panda/UR5 TimeOpt cohort and complete measurement driver exist. Full torque/torque-rate certification needs separate rigorous dynamics enclosures.

## Independent recovery package

See [recovery gates](docs/RECOVERY_PLAN.md), [coding-agent handoff](docs/AGENT_RECOVERY_PROMPT.md), `boundary_reference.py` and `exact_k3_reference.py`. These dependency-free exact-rational implementations are analytic reference tools and do not establish native parity or campaign measurements. `BOUNDARY_PASS` alone is not a full K3 certificate.
