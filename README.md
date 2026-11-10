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
