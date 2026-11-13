# Coding-agent execution handoff — native certification + proof laboratory

You are working in \`pkicki/curobo_certify\`. Read \`AGENTS.md\`, \`docs/CAMPAIGN_PLAN.md\`, \`docs/PROGRESS.md\`, \`docs/COMPILED_HULL_AND_NATIVE_LINKAGE_PLAN_20261111.md\`, and \`docs/INTERACTIVE_BERNSTEIN_CAMPAIGN_20261113.md\` in full before modifying anything.

**GOAL:** produce a rigorously justified, native-linked K3 continuous-time certificate for unchanged cuRoboV2-Fast TimeOpt solutions, implement a genuinely faster *complete verified* certifier, and measure H0/uniform/adaptive performance on frozen real paths. In parallel, make the new interactive exact-rational Proof Lab usable with real residual bundles and high-quality visual diagnostics. Do not confuse mathematical demo results with native certificates.

## Important branch/evidence warning

The published Proof Lab branch is \`exp/interactive-bernstein-proof-lab-20261113\` based on remote \`c01d1b6\`. Previous local user reports reference an unpublished integration branch \`exp/timeopt-compiled-hull-integration-20261111\` at \`1100caaa3f077d30c40aef393f4390c59670f4a5\` with eight real Panda source outputs, 1,176 native trace matches and real CPU profiling. This SHA is NOT fetchable on public GitHub at the time of authoring. First inspect all local branches, ignored \`runs/\`, untracked files, manifests and source pins. If local integration exists, create a NEW child branch from it and cherry-pick the Proof Lab work deliberately after examining ancestry. If missing, do not fabricate those earlier experiments. Do not reset, force-push, merge, delete evidence or create PRs.

Native cuRobo pin is \`c301cb00a87012c2808b1a60649c4065c9b3c95a\`. Keep original upstream and TimeOpt outputs unchanged. If needed, apply source-instrumentation diffs only to a disposable pinned clone.

## First 10 actions (do these before GPU campaigns)

1. Record exact Git branch/HEAD/status, Python/Node/GCC/NVIDIA driver/CUDA/Torch/cuRobo versions, hardware, source hashes, affinity and compiler floating-point flags. Preserve ignored files.
2. Inspect and test new \`web/proof_core.js\`, \`web/proof_app.js\`, \`web/proof_template.html\`, \`scripts/build_proof_playground.py\`, \`scripts/export_playground_residuals.py\` and tests. Commit fixes, never suppress failing assertions.
3. Run \`node --test tests/*.test.cjs\`, \`python -m unittest discover -s tests -p 'test_*.py' -q\`, \`python -m compileall -q src scripts tests\`, \`python scripts/build_proof_playground.py\`. Hash/check generated HTML against committed report.
4. Export the analytic fixture: \`python scripts/export_playground_residuals.py fixtures/boundary_multispan_analytic.json --span-index 0 --joint 0 --out /tmp/analytic_residuals.json\`. Open/import it in offline HTML. Verify exact rendered coefficients and statuses against Python Fraction reference.
5. Browser acceptance in Firefox and Chromium: no console exceptions, keyboard and button controls work, mobile view legible, animation/step/reset functional, histogram/bounds stable, JSON imports fail closed. Save two screenshots for report (do not commit sensitive data).
6. Add adversarial browser/cross-language tests: rational exponent/limits, strict interior roots, near tangencies, genuine witnesses, high-degree inputs, malformed JSON/large inputs, depth/node budgets, missing/modified leaves, zero interval, coefficient overflow to plot float. Production browser-only tests must be distinguished from the mathematically independent backend verifier.
7. Reconcile source/provenance and re-establish reported eight Panda models; verify native output hashes/durations are unchanged.
8. Run reproducible native operand diagnostics, first **same native progress s**, then same physical t/time inversion; preserve registered scaled-relative jerk tolerance 4e-4.
9. In parallel prototype a compiled **full** K3 certificate producer AND separate independent validator. Do not call \`kinematic_hull.cpp\` full certifier before boundary, positivity, knot regularity and independent replay are integrated.
10. Freeze a source-query manifest for additional real hard cases, but do not open Dev40/Holdout100/Confirm300 until the correctness gates.

## Required engineering details

**Native linkage:** compare and analyze q',q'',q''',x,x',x'', path progress, knot-side selectors and FMA order; 1,176 bitwise arithmetic replays reported, but NOT a rigorous error bound. Derive defensible all-time enclosure for FP32 coefficient generation/evaluation and s(t) inversion or report BLOCKED_NATIVE_LINKAGE. Never accept empirical max difference as a proof or quietly relax parity.

**Compiled pipeline:** interval coefficient enclosures and de Casteljau, typed fail-closed UNKNOWN, strict x positivity and finite time traversability, complete endpoints/continuity/jerk constraints, proof leaves and complete-independent verification, source/limit/model hashes. At least 10k seeded multijoint cross-checks per test configuration against exact Fraction; MPFR directed rounding if available. Test compiler/architecture/fenv/FTZ assumptions; zero observed under-enclosures; no universal proof from tests alone.

**Real cases:** original Panda output paths (8 baseline + exploratory varied tasks), physical actuator limits, more high-utilization and challenging obstacles/payloads. Keep original TimeOpt geometry, 32 timing controls, endpoint contract, limits, duration and arrays untouched. Seek initial-hull UNKNOWN but exact peak-proved-feasible residuals. Preserve all submitted/native-success/K3-eligible denominators. UR5e only after model/limits validated.

**Benchmarks:** match workloads and report hull-only, CPU full producer+independent replay, GPU/hybrid frozen-only, and fresh native total-added latency separately. H0 vs U1/U2/U4/U6 vs adaptive vs no-witness; depths 0/2/4/8/12/16; nodes 2048/8192/32768; S8193/S32768 diagnostic only. 50 warmups, >=10 paired balanced AB/BA repeats, raw per-path data, source-cluster bootstrap. RTX3090 primary, A100 after code freeze. Original targets K3>=99% eligible, fresh added p50<=1ms, p95<=3ms are goals NOT evidence.

**Visualization:** add real residual bundles and prove local single-residual bounds (not native), linked original TimeOpt q/x and constraints, clickable span/joint/family/leaf tree, exact peak/hull gap, violation and unknown labels, native-vs-model samples with caveats, and report latency/leaf-cost Pareto plus ECDF. Keep existing Proof Lab fully offline and functional. Add meaningful accessibility/browser tests and report artifact SHA.

## Stop / failure rules

- Fail native certification if any unsupported dynamics is presented as supported, any required family/interval not proven, or any native error cannot be defensibly enclosed.
- Stop on any false certificate, coefficient under-enclosure, parity drift in pristine-vs-trace outputs, model mutation, or broken source provenance. Diagnose, fix, and rerun affected gate before proceeding.
- Treat node/time/depth exhaustion as UNKNOWN, never VIOLATED or CERTIFIED.
- If native linkage is blocked, continue fully labeled exploratory exact-model speed/visualization work but DO NOT open formal native certification cohorts or report fabricated pass counts.
- No pull request, merge, force-push or changes to \`main\`.

## Work style and final handback

Work autonomously: inspect -> implement -> tests -> investigate/fix -> rerun -> benchmark -> commit locally and push your dedicated experiment branch (not main). Commit after logical passing work packages. Update \`docs/PROGRESS.md\` continuously with exact commands, hashes, scopes, dates and numerical denominators. Preserve historical raw artifacts.

At completion give: branch/HEAD/commit SHAs, source/repo SHA and hardware/environment, tested commands/passes/failures, browser screenshot + interactive local HTML paths, exportable genuine real residual bundle, proof status by family/path (MODEL vs NATIVE), native arithmetic-linkage findings, exact demo counterexamples, compiled full-vs-hull-only costs, paired latency p50/p95/p99/CIs for each method on *matched* workloads, split/leaf/margin/hull-gap distributions, submitted and eligible counts, duration/hash invariance, unresolved blockers and GO/NO-GO decision. A blocked or negative result is a valid outcome if supported by evidence.
