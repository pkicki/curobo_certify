# Recovery campaign — unchanged TimeOpt + adaptive certificate

Date 2026-11-10. **Status: recovery only; no native certification or measured GPU performance.**
This recovery plan supplements [the preregistered campaign](CAMPAIGN_PLAN.md); it does not loosen its gates.

## 0. Branch and evidence integrity

The user reports an *unpublished local* branch `exp/timeopt-adaptive-certification-20261110` at `ece5a3a`, with partial P0–P3. The GitHub remote did not contain this branch at inspection; the published campaign-plan parent is `ef5f32033bd4690484cb882b3f0372d460f423f9`. The independent recovery branch must therefore **not** be mistaken for a successor of `ece5a3a`. On the development machine, create an integration branch from the local unpublished HEAD and cherry-pick the recovery commit(s), preserving previous commits, files and evidence. No reset, force-push, PR or merge to main.

The earlier execution report said Python 3.12.4, no importable PyTorch/cuRobo or Node, nearby cuRobo `83362824…` rather than required `c301cb00a87012c2808b1a60649c4065c9b3c95a`, zero real cohort paths, unresolved boundaries and native parity. Those are **reported**, not independently reproduced.

## 1. Scientific contract (invariant throughout)

Question: **how much *end-to-end* latency is added to an unchanged TimeOpt solution by rigorous adaptive subdivision alone?** Never change the geometry, 32 timing controls, endpoints, limits, planning margins or original duration. Exact de Casteljau subdivision partitions residual representations; it is not retiming or repair. Preserve before/after input hashes.

Distinguish `BOUNDARY_ONLY`, proof-backed `K3_RIGOROUS`, `SAMPLED_DIAGNOSTIC`, and torque/torque-rate `UNSUPPORTED`. Never claim full certification if boundary/positivity/knot checks remain UNKNOWN or dynamics is unsupported. A dense overshoot is a diagnostic, not a rigorous violation witness.

The independent `src/curobo_certify/boundary_reference.py` is a reference **sub-contract** checker using exact Fractions: local-power cubic q(s), x(s), exact C2(q), C1(x), explicit q/x endpoint targets, declared endpoint accelerations, strict-rest acceleration=0, and local endpoint duration integrability. Local `BOUNDARY_PASS` does not establish interior positivity or K3. Binary floats are interpreted exactly, so differently quantized coefficients can fail exact equality. Native FP32 kernel parity is a separate prerequisite. Replay protocol: JSON object with `schema_version:1`, `basis:"local_power_ascending_u"`, `spans:[{left,right,q:[[a0,a1,...],...],x:[b0,b1,...]}]`, and `targets:{q_start,q_end,x_start,x_end,strict_rest?,acc_start?,acc_end?}`. Coefficients use u=(s-left)/(right-left). Use rational strings for non-dyadic exact values. Run `python scripts/replay_boundary.py input.json --out output.json`; output SHA256 identifies original input and both full K3/full-contract statuses are always NOT_EVALUATED.

## R0: pinned environment and exact native adapter (STOP until PASS)

1. Save `git status`, `git log`, `git branch -avv` and all upstream SHAs. Use isolated Python env/conda and a separate clean cuRobo checkout pinned to `c301cb00a87012c2808b1a60649c4065c9b3c95a`. Do not modify the nearby checkout. Record compatible Python/PyTorch/CUDA/NumPy/SciPy versions, CPU/GPU, driver, power/clocks, node and native compilation settings. Run upstream smoke.
2. Extract actual effective q(s) geometry including boundary modifications, union knot topology, coefficient interpretation and dtype; time scaling spline x(s) from exactly 32 timing controls; original and scaled limits, model/payload, endpoints, native duration, status and original array SHA256. Borrowed cuRobo result memory must be cloned before another solver call. Never fit polynomials to sampled states.
3. Native-adapter parity: 1,001 deterministic random interior points **plus both sides of every knot** for q, q_s, q_ss, qdot, qddot, jerk; explicit dtype-based numerical thresholds and full discrepancies. Check stored outputs after a second solve to detect aliasing. Fail `BLOCKED_ADAPTER` on any unknown field. Parity is numerical evidence, not itself a formal enclosure.

## R1: prove all piecewise obligations

1. Convert real native geometry/time splines to local-power coefficients on the union knot partition; multiply derivatives by h^-k to express them in global s. No arbitrary continuity tolerance can be introduced into a strict mathematical contract.
2. Integrate the exact boundary reference with P1/P2 interval/Bernstein scaffold from the local branch, retaining scoped `BOUNDARY_PASS/UNKNOWN/VIOLATED`. Verify q through second derivative, x through first derivative at knots, one-sided jerk, each declared endpoint state, original vs strict-rest acceleration conditions, and local integrability for zero-speed endpoints. Internal stationary knots require separately justified dynamics/traversal. Investigate native FP32 evaluation vs exact mathematical coefficients.
3. Compile residual polynomials: positivity -x; velocity q_i'^2 x / vlim_i^2 - 1 (degree <=7); acceleration +/- (q_i'' x + 0.5 q_i' x') / alim_i - 1 (<=4); jerk x(q_i'''x + 1.5q_i''x' + 0.5q_i'x'')^2/jlim_i^2 - 1 (<=9). Use outward-rounded interval coefficient operations and exact midpoint de Casteljau subdivision. Distinguish strict interior positivity, finite traversal, invalid limits, rigorous witness VIOLATED, `UNKNOWN_DEPTH/NODES/TIME/BOUNDARY`, and dynamics `UNSUPPORTED`. CPU exact Fraction oracle and independent proof replay must agree on a deliberately limited analytic subset. No full `CERTIFIED` unless *every* K3 inequality and boundary condition is proven.
4. Persist full proof leaves/interval upper bounds, split ancestry and source hashes. A tiny independent checker must replay leaves and conjunctions rather than trusting a serialized success flag.

## R2: adversarial tests / acceptance of first real certificate

Run at least 200 seeded analytical/property/adversarial tests: hidden positive residuals, non-dyadic tangencies, g=[-1,0.2,-1], exact rational polynomial extrema, continuity jumps q/q'/q''/x/x', strict-rest acceleration, zero-speed endpoints (linear finite vs quadratic infinite), internal zero x, tiny spans, large dynamic range, coefficient NaN/Inf, degree/limit mismatch, proof node/time exhaustion and input mutation. Real native certificate must pass proof replay with immutable coefficients and native parity; 8,193 and event-aware 32,768 samples are *independent falsification diagnostics*, not proofs. Stop on any observed false certificate. Only then claim the first fully proven K3 native trajectory.

Commands for standalone reference checks:
```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m compileall -q src scripts tests
python scripts/replay_boundary.py path/to/explicit_fixture.json
```
These tests have no GPU dependencies and do not certify native outputs.

## R3: freeze real input populations

After R0/R1: Panda/UR5, each Dev20/Holdout50/Confirm150 (pooled Dev40/Holdout100/Confirm300), stratified by scene/query before related variations, seed 20261110. Preserve the *full* denominator: submitted query, native success/failure, adapter valid, K3 eligible, per-family proof state, full-contract state. Hash/record source config, geometry, limits, controls, model, endpoint contract and duration. No invented paths. Keep private dataset bytes out of public Git.

## R4: first measurements on Dev40

On identical frozen outputs evaluate H0 (no split), U1/U2/U4/U6, adaptive breadth-first A, A-no-witness, plus sampled diagnostics S8193 and event-aware S32768. Depth sweep 0/2/4/8/12/16, node limits 2048/8192/32768, selected deadline curves only after deterministic budgets. Show fraction proven without split, leaf/depth distribution, rigorous violations vs unknowns, and the growth of certificate size.

Profile native output clone/export, common knot extraction, residual construction, outward polynomial bounds, splitting/queue, transfer, synchronization and mandatory proof replay. Benchmark *both* frozen-output cost and **complete fresh-path wall latency** from device-ready input to CPU-visible certified decision, including every necessary proof task. Fifty warmups, 10 paired AB/BA randomized repeats per path; report p50/p90/p95/p99/max and paired 10,000-source-cluster bootstrap 95% CI; do not claim p99 precise on small distinct-path counts. Optimize only measured bottlenecks after CPU reference passes. RTX 3090 primary, A100 replication later; no cross-GPU method comparison.

## R5: lock, holdout, report and science decision

Lock code/config/hashes *before* Holdout100; if correctness fails, diagnose and use a newly untouched holdout for final qualification or label a rerun exploratory. Confirm300 and second GPU only when gates pass. Extend offline interactive HTML with true q/x/qdot/qddot/jerk, signed and squared residuals, knot-side markers, native sampled points, rigorous leaf enclosures, adaptive-tree replay, duration, outliers, ECDF, overhead-vs-leaves, waterfall and source SHA. All analytical/demo/sampled/rigorous data must be visibly separated. Run Node tests and offline Firefox/Chromium console/filter/export tests. Never use browser math as a proof.

Prerecorded initial usefulness targets for K3 on physical limits (not results): >=99% of eligible native successes certified, added fresh-path p50 <=1ms and p95 <=3ms on RTX 3090, with hard zero observed false-certificates and no mutation. Full torque/rate remains separate. Section 8 decision is `NOT_REACHED` until the actual studies are done; then choose PASS_BASELINE / CORRECT_BUT_SLOW / INCONCLUSIVE_DOMINATED / VIOLATION_DOMINATED / BLOCKED_DYNAMICS.

## Per-package and final evidence

For every package: minimal implementation → meaningful tests → fix failures → rerun → commit → append `docs/PROGRESS.md` with literal commands, paths, counts and next gate. Final handback: branch/SHAs, clean status, exact environment, adapter parity, frozen cohort hash ledger, per-family certified/violated/unknown with denominator, K3 vs full statuses, latency p50/p95/p99/CI or NOT RUN, original-hash/duration invariance, top witnesses, plot/export report path, and section 8 decision or NOT REACHED. **No PR, no merge.**
