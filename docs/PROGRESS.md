# Campaign progress

2026-11-10 — Planning/report preparation. Repository was empty at inspection.

Completed: upstream API inspection at c301cb00a87012c2808b1a60649c4065c9b3c95a; detailed staged plan; GPT-6 Luna handoff; offline report and analytic subdivision demonstration; compact measured-data import contract.

Not run: TimeOpt GPU integration, real-path certification, GPU latency experiments, production interval arithmetic, torque/rate enclosures, Holdout100 and Confirm300. Every experimental statistic remains NOT RUN.

Next action: P0 environment and source contract. Check the exact native geometry and borrowed output lifetime before implementing residual extraction. See docs/LUNA_PROMPT.md.

Validation of planning deliverable: nine Node tests passed (curve preservation, polynomial conversion/derivatives, analytic outcomes, physical residual equations, paired aggregation and input validation, plus report initialization and JSON import with a DOM test double); generated HTML scripts parse. Full browser rendering could not be checked in this environment: local Chromium cannot open its process socket and the cloud browser blocks local-file URLs. Visual/browser acceptance remains for Luna; no claim of completed browser QA.

2026-11-10 — P0 environment/source-contract vertical slice (blocked at GPU parity gate; commit records this evidence). Added environment capture, pinned-source adapter guard, borrowed-output clone helper, and an optional dependency declaration for the requested upstream SHA. The inspected local cuRobo checkout is at 83362824f6321af92429cc46b1af011b1d533ca4, not the required pin, and has unrelated dirty files; it was not modified. The required SHA is not in its object database. Current Python 3.12.4 has NumPy 1.26.4/SciPy 1.13.1, but no PyTorch/cuRobo; Node is absent. RTX 3090 and A100 are visible, but clocks show idle/non-campaign state. `python scripts/capture_environment.py --out runs/environment.json` captured these facts. `python -m compileall -q src scripts` passed. Adapter parity and real-path smoke are NOT RUN; effective geometry export remains BLOCKED_ADAPTER. No input cohort exists. Do not proceed to corpus capture or GPU claims until exact source/dependencies are available.

Next action: P1 rigorous CPU interval/Bernstein reference work, independent of upstream/GPU availability; retain P0 parity blocker.

2026-11-10 — P1 CPU reference arithmetic vertical slice. Added binary64 outward interval primitives, exact-rational endpoint conversion, power/Bernstein conversion, global-span derivative scaling, Bernstein products/elevation and de Casteljau subdivision, plus conservative status aggregation. The arithmetic assumptions are stated in `docs/CPU_INTERVAL_ASSUMPTIONS.md`. `python -m unittest discover -s tests -p 'test_*.py' -v`: 10 tests passed, covering rational/dyadic enclosures, polynomial product/derivative identities, hidden-peak subdivision (`[-1, 0.2, -1]`), non-dyadic tangency remaining inconclusive, zero divisors, NaN and indeterminate infinities. This is a CPU arithmetic slice only: no K3 certifier, independent proof replay, platform conformance audit, or native inputs; no CERTIFIED outcomes are produced. Node report tests remain unavailable (Node absent).

Next action: implement P2 residual compilation and a separate replay checker; strict moving-interior positivity and endpoint/regularity semantics must be implemented before any K3 CERTIFIED status is possible.
