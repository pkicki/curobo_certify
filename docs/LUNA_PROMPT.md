# GPT-6 Luna implementation prompt

You are implementing the TimeOpt adaptive-certification campaign in pkicki/curobo_certify. Read README.md, AGENTS.md, docs/CAMPAIGN_PLAN.md and docs/PROGRESS.md first. The current deliverable is a plan plus an educational HTML demo; no real GPU results exist yet. Do not interpret demo values as benchmark data.

Implement P0 through P7 sequentially on exp/timeopt-adaptive-certification-20261110. Make one coherent commit per completed package and update PROGRESS.md. Do not create a PR or merge. Keep the native TimeOpt solver, its 32 timing coefficients, planning margins, endpoints and geometric path unchanged in the primary campaign. Exact certificate subdivision must not become a new optimization problem.

Use pinned NVlabs/curobo c301cb00a87012c2808b1a60649c4065c9b3c95a after checking compatibility. Preserve attribution; prefer a dependency and an adapter. Trace the actual effective geometry, clone borrowed outputs when needed, and freeze content hashes. Run the adapter parity gate before building the certifier. Record missing GPU/data/dependencies explicitly; continue independent CPU/reference/report tasks, but do not substitute made-up values.

Implement and justify outward-rounded CPU polynomial certificates first. Build a separate proof checker. CERTIFIED means all requested obligations are enclosed, including boundary/regularity/positivity requirements. A loose upper hull, a dense check, high precision alone or FP64+epsilon is not a proof. Use UNKNOWN for unresolved cases. Start with K3 (velocity, acceleration, jerk); torque/rate need separate rigorous dynamics enclosures and remain UNSUPPORTED until those exist. Keep K3 and full-contract status distinct.

Freeze Dev40/Holdout100/Confirm300 by source query/scene, as in the plan. Never tune on holdout. Evaluate no-subdivision, fixed uniform and adaptive subdivision on identical TimeOpt outputs; report native and dense sampled baselines honestly. Measure frozen-output overhead and synchronized fresh-path complete-request overhead. Include extraction, transfers, decisions and required proof work; separate startup, cache reuse, instrumented profiling and trace collection. Use paired repeats and source-cluster uncertainty; retain all failures and timeouts.

After Dev40, report bottlenecks before GPU acceleration. Do not port a non-rigorous numerical screen and call it a certificate. If necessary proof checking still runs on CPU at request time, include its cost. Re-run correctness gates after each arithmetic change.

Extend reports/timeopt_certification_campaign.html with measured data, per-path retiming/constraint plots, knot/sample/enclosure overlays, adaptive-tree replay, outlier links, timing statistics and exports. Follow the report acceptance criteria; keep it self-contained and runnable offline. Mark every source of data and unsupported family. Do not implement repair or certificate-constrained optimization in this campaign.

For each package: state the goal, implement a minimal vertical slice, run its meaningful tests, inspect failures, fix bugs, rerun affected checks, commit, update progress, then continue. If a mathematical assumption is unresolved, write its exact statement and a failing/minimal example and leave the affected claim blocked. Never weaken acceptance criteria to make tests pass.

Finish with: commit/branch; environment and exact input cohort; correctness/proof scope; certified/violated/unknown counts; added latency p50/p95/p99 with uncertainty and full denominators; stage bottlenecks; duration/input invariance; top failure examples; interactive HTML path; decision under section 8; remaining blockers. Report negative results as carefully as positive ones.
