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
