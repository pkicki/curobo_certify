# Campaign progress

2026-11-10 — Planning/report preparation. Repository was empty at inspection.

Completed: upstream API inspection at c301cb00a87012c2808b1a60649c4065c9b3c95a; detailed staged plan; GPT-6 Luna handoff; offline report and analytic subdivision demonstration; compact measured-data import contract.

Not run: TimeOpt GPU integration, real-path certification, GPU latency experiments, production interval arithmetic, torque/rate enclosures, Holdout100 and Confirm300. Every experimental statistic remains NOT RUN.

Next action: P0 environment and source contract. Check the exact native geometry and borrowed output lifetime before implementing residual extraction. See docs/LUNA_PROMPT.md.

Validation of planning deliverable: nine Node tests passed (curve preservation, polynomial conversion/derivatives, analytic outcomes, physical residual equations, paired aggregation and input validation, plus report initialization and JSON import with a DOM test double); generated HTML scripts parse. Full browser rendering could not be checked in this environment: local Chromium cannot open its process socket and the cloud browser blocks local-file URLs. Visual/browser acceptance remains for Luna; no claim of completed browser QA.


## 2026-11-10 — Independent recovery branch

GitHub remote did not contain user-reported local `exp/timeopt-adaptive-certification-20261110` at `ece5a3a`. The independent recovery branch starts from the published `ef5f320` campaign plan. It cannot verify or replace the local P0–P3 implementation; coding agent must reconcile local ancestry before cherry-picking.

Implemented a dependency-free exact-Fraction **boundary-only** oracle (`src/curobo_certify/boundary_reference.py`), explicit schema/replay CLI (`scripts/replay_boundary.py`), and standalone regression tests. Verified locally in Python 3.13.5: **17 tests passed**, `compileall` clean. Files uploaded with matching Git blob hashes for tested source and test files. This is not a native TimeOpt adapter, K3 certifier, or GPU result. No corpus, certificates, native parity, or latency measurements.

Next gate: recover pinned cuRobo environment and native adapter parity, then integrate with local unpublished CPU interval/Bernstein scaffold. See `docs/RECOVERY_PLAN.md` and `docs/AGENT_RECOVERY_PROMPT.md`.

Follow-up validation: strengthened malformed-JSON fail-closed replay and negative internal knot tests. Python 3.13.5: 19 reference tests passed, compileall clean. Native adapter/GPUs still NOT RUN.

### Exact rational K3 reference package — analytic only

Added `src/curobo_certify/exact_k3_reference.py` with global-s derivative scaling, exact polynomial K3 residual construction, exact power-to-Bernstein conversion, exact de Casteljau subdivision, explicit violation witnesses, depth/node budget UNKNOWNs, strict moving x checks, and leaf completeness/coefficient replay. Added `scripts/replay_k3_reference.py`, two test files and a two-span rational fixture. Local Python 3.13.5: **37 tests passed**, compileall passed, including 256 seeded exact-rational spot-check cases in one property test. Analytic two-span fixture: CERTIFIED_K3_REFERENCE, 10 proof leaves, replay PASS; native equivalence and torque/rate remain NOT EVALUATED/UNSUPPORTED. An independent interval arithmetic checker and actual native TimeOpt parity remain mandatory; the reference replay reuses polynomial building and is not a separately verified numerical implementation. No real paths, no GPU latency.


## 2026-11-11 — Jerk instrumentation and Bernstein-tightness recovery (independent branch)

From published f40d82a, implemented:
- Opt-in pinned-cuRobo CUDA trace patch *generator*, guarded by exact source revision/blob and no source mutation, and binary32 trace decoder.
- Exact Fraction branch-and-bound residual peak brackets; initial Bernstein hull overestimation interval and rigorous mathematical-model margin intervals, budget failure semantics.
- Frozen-model CPU hook timing with explicit EXPLORATORY_FROZEN_MODEL_CPU_ONLY scope and raw repeats; default exact-reference hook includes parsing and replay, but is not the local outward-rounded CPU implementation.
- Offline self-contained HTML tightness plot/table generator with SHA and per-family filters.

Validation here: **57 local Python tests passed**, compileall clean, Node 22 JavaScript syntax check on generated report passed. Analytic two-span fixture produced 10 per-family/positivity residual records and an offline HTML visualization. No real Panda files in the independent environment; no CUDA build, source instrumented trace, native correctness parity, Dev40, native-linked K3 count, or real-path timing was run.

The local user's newest integration at 018fa07c remains unpublished; agent must reconcile it before using these modules on real GPU paths. **Section 8: NOT REACHED.**
