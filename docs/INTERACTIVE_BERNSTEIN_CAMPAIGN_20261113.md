# Native TimeOpt certification + interactive Bernstein proof campaign

**Prepared 2026-11-13.** Scope: certify unchanged cuRoboV2-Fast TimeOpt output and determine incremental end-to-end latency. Research first; preserve all existing source outputs and source provenance. This document is an *implementation and experiment plan*, not evidence that the planned native/GPU trials ran.

## 0. Evidence ledger, invariants, and current branch

- Published base: `exp/timeopt-compiled-k3-hull-linkage-20261111`, `c01d1b6055eebd5ecd1dd95b8926fb1a4383cf11`.
- New independent branch: `exp/interactive-bernstein-proof-lab-20261113`; it contains an exact-rational single-residual interactive playground, not a GPU certifier.
- Previously *reported but absent on GitHub*: integration branch `exp/timeopt-compiled-hull-integration-20261111` at `1100caaa3f077d30c40aef393f4390c59670f4a5`; earlier local `54e3905ec058f7f060eb3ea5b908e81e63f895e9`. These reported commits must be inspected on the real development workstation. NEVER assume that the independent published branch includes their evidence or overwrite them.
- Native upstream source pin: `NVlabs/curobo@c301cb00a87012c2808b1a60649c4065c9b3c95a`.
- Previously *reported*, not replicated here: 8/8 frozen Panda mathematical K3 proofs; 0/8 registered native jerk parity passes; 1,176 native trace replays matched final arithmetic; C++ hull-only p50 84.72 ms vs full Python+replay ~1.457 s on eight real frozen outputs; native initial-hull component ~2.8 ms; no formal Dev40/Holdout100/Confirm300. The source machine must re-establish these values with manifests.
- No PR, merge, force-push, branch deletion, input mutation or change to original durations. The optimization problem, TimeOpt controls (32), geometry, physical/planning limits, endpoints, payload and native solver result must remain unchanged. Adaptive Bernstein subdivision is *only a representation refinement*.
- Distinguish EXACT_ANALYTIC_SINGLE_RESIDUAL, EXACT_SERIALIZED_K3, FROZEN_OUTPUT_EXPLORATORY, NATIVE_LINKED_K3, SAMPLED_DIAGNOSTIC and UNSUPPORTED_DYNAMICS in all files and UI.
- Do **not** relax registered jerk scaled-relative parity threshold 4e-4 to fabricate a pass. If using a different rigorous native-evaluation enclosure, specify and separately justify the new mathematical implication and preserve 4e-4 as its original regression metric.

## 1. Questions / predeclared hypotheses

Q1. Does the extracted exact q(s), x(s) representation conservatively enclose native CUDA evaluation at every physically requested time, including knot sides and time inversion?

Q2. For actual feasible but hull-inconclusive outputs, does adaptive splitting yield fewer leaves and less *complete certificate* latency than uniform U1/U2/U4/U6? H1: adaptive refinement improves the work/coverage Pareto frontier in this population; no guarantee if initial hulls already pass or the true trajectory violates limits.

Q3. Can a compiled conservative interval builder **plus independent proof replay** lower *fully verified* CPU certificate costs by >=10× compared with full Python reference on paired frozen paths, without changing any correctness contract? H2: moving proof construction and replay matters more than optimizing the ~2.8 ms hull kernel alone. This is a hypothesis, not an achieved result.

Q4. Can native K3 certification add p50 <=1 ms and p95 <=3 ms CPU-visible end-to-end on RTX3090 while achieving >=99% certification of eligible native successes? These are original ambitious acceptance targets, not measurements; compare against observed latency as well, even if far from target.

Q5. Which residual families, margins, depths, knot conditions and native operand stages cause failure, inconclusiveness or expensive proof trees?

## 2. Committed deliverable V0: playable proof lab

Files:
- `web/proof_core.js`: exact BigInt rational input, de Casteljau splitting, interval Bernstein hulls, strict interior-zero checks, exact violation witnesses, explicit depth/node unknown, leaf-coverage/coefficients structural replay, global peak and initial-hull-gap brackets.
- `web/proof_app.js` and `web/proof_template.html`: responsive animated plot, proof tree, controls, slider-based point probe, import/export.
- `scripts/build_proof_playground.py` -> `reports/bernstein_proof_lab.html`: self-contained offline HTML artifact.
- `scripts/export_playground_residuals.py`: export per-span/per-family/per-joint exact residuals from the registered serialized cubic q/x mathematical schema.
- `tests/proof_core.test.cjs`, `tests/test_proof_playground.py`: adversarial arithmetic, witness, proof budget, replay tampering, report reproducibility, exporter behavior.

### Quickstart

From repository root:

```bash
python scripts/build_proof_playground.py
node --test tests/proof_core.test.cjs
python -m unittest discover -s tests -p 'test_*.py' -q
python -m compileall -q src scripts tests

python scripts/export_playground_residuals.py \
  fixtures/boundary_multispan_analytic.json \
  --span-index 0 --joint 0 --out /tmp/analytic_residuals.json
```

Open `reports/bernstein_proof_lab.html` locally (no server), click **Import JSON examples**, choose `/tmp/analytic_residuals.json`. This JSON is intentionally scoped to *one exact residual per item*, not whole-trajectory certification. The browser's structural proof replay uses the SAME rational arithmetic; it must never be described as an independent proof checker.

Preset lessons:
1. `[-1, 2/5, -1]`: initial hull upper +2/5 but true peak -3/10; depth 0 UNKNOWN, one split CERTIFIED.
2. `[-1, 2, -1]`: positive exact point witness at s=1/2; genuinely VIOLATED.
3. `[-1/4, 1/4, -1/4]`: tangent zero; feasible non-strict, invalid for strict x>0 interior.
4. Degree-nine synthetic jerk-like residual: more Bernstein control points than physical constraints; not automatically a real robot jerk profile.
5. All-negative coefficients: depth-zero certifiable fast path.
6. Zero polynomial: acceptable for ≤0; strictly negative condition fails.

For a residual `r(s)=sum_k b_k B^n_k(s)`, `min(b) <= r(s) <= max(b)` for every s on a leaf. A positive `max(b)` is *inconclusive*, not a violation. The global maximum bracket in the lab is an exact mathematical bound built from dyadic leaf hulls and exact point witnesses; the smooth plotted curve is a float diagnostic.

## 3. P0 — reconcile local branches, source pins and reproducibility

On the GPU development workstation:

1. Record `git status --short`, `git log --all --oneline -25`, `git branch -avv`, HEAD and upstream remotes for this repo and cuRobo. Preserve untracked/ignored `runs/` data and user work.
2. Find the *actual local* integrated baseline, reported at `1100caaa`; verify its ancestry, worktree state, branch, compiled arithmetic validation and saved eight Panda frozen outputs.
3. If that local branch exists, create a **fresh child branch** and selectively cherry-pick independent proof-lab changes, resolving collisions without replacing local integration. Do not reset the original branch. If it does not exist, stay on this public development branch and record unavailable evidence; do not simulate GPU inputs.
4. Verify pinned cuRobo source and local build SHA, environment, compiler flags, CUDA, GPU, floating point environment, CPU affinity, backend and memory ownership (borrowed cuRobo outputs must be cloned before reuse).
5. Run Python+Node suite, build HTML, inspect it in Firefox/Chromium both desktop and mobile viewport. No remote scripts, blocked CDNs, JS errors, or broken import/export. Save screenshots, browser console log and hash of HTML.
6. Freeze immutable source-model and output SHA256 pairs and record duration and original limit hashes.

**Gate P0:** know the true parent and source identity, no source loss, all existing tests pass, and the demo runs offline. If not, stop and report blocker.

## 4. P1 — exact native representation and continuous FP32 linkage

The current chief blocker is native FP32 jerk mismatch. Work on diagnostic copies ONLY.

### P1a: stage-wise native arithmetic attribution

- Reuse the 1,176 reported device captures and generate more targeted traces covering knots, very low x, tangencies, large q''' and x'', high jerk utilization, zero-speed endpoints, and time inversion. Verify uninstrumented vs instrumented native output bitwise on the same inputs.
- Compare native q', q'', q''', x, x', x'', s, t and jerk bitwise/ULP/abs/scaled-rel. Run two modes: **same native s** (no inverse-timing difference) and same **requested physical t** (including native s inversion).
- Reconstruct exact native Horner/FMA order using the pinned source and check if operand-level replay is bitwise. Examine derivatives on both sides of every knot. Report output lifetime, CUDA graph on/off, kernel settings and compiler reproducibility.
- Use `scripts/attribute_native_jerk.py` for diagnostic geometric-vs-timing decompositions, but its effects are NOT rigorous error bounds.

### P1b: rigorous native evaluation enclosure

- Model FP32 controls/coefficients exactly as stored. Establish conservative bounds on conversion/extraction, cubic/spline evaluation, derivative construction, time inversion and final jerk arithmetic for **every** relevant s or t in the continuous interval.
- Prefer explicit interval Taylor/derivative error propagation with correct directed rounding/MPFR independent audit; use segment-level monotonicity to bound any inversion branch. Model FP32 FMA/rounding and underflow, exceptional operands and knot-side branches.
- Test at least 1,001 independent random internal points **plus both sides of every knot**, then S8193 and event-aware S32768 as falsification diagnostics. Validate on diverse paths and limits, not only the eight easy Panda paths.
- Require independently reviewable and replayable bounds on native physical residuals. A finite maximum observed difference (including 0.00368 jerk) does NOT establish a continuous-time bound.
- If a complete numerical model cannot be justified, explicitly set `BLOCKED_NATIVE_LINKAGE`; do not grant native certificates by changing tolerance.

**Gate P1:** first mathematically defended native K3 certificate, with all families, endpoints, knot obligations, strict interior positivity and native linkage accepted; else all native claims remain blocked.

## 5. P2 — compiled full K3 proof construction and replay

Current C++ `kinematic_hull.cpp` implements initial-hull screening ONLY. Extend it without weakening semantics:

1. Define stable C ABI for exact serialized q/x common spans and per-joint original limits, with validated dimensions and runtime caps; return per-family statuses and compact dyadic proof leaves rather than just a maximal coefficient.
2. Preserve global-s derivative scaling and correct family definitions: positivity -x, velocity q'^2 x/v²-1 (degree up to 7), ±acceleration ((q''x + q'x'/2)/a) with sign (degree up to 4), jerk x(q'''x+1.5q''x'+0.5q'x'')²/j²-1 (degree up to 9).
3. Use outward enclosures for every polynomial/Bernstein operation. Adaptive midpoint de Casteljau with deterministic BFS, strict node/depth/time limits; unknowns fail closed. Include explicit rigorous witnesses with rational/interval checking, not inferred from positive hull coefficients.
4. Integrate complete endpoint/regularity checks: q C², x C¹, one-sided jerk, endpoint q/x/acceleration contracts, strict interior x>0, finite local time integrability near x=0, true internal zeros and knot-side evaluation.
5. Produce compact replayable leaf certificates: source/output SHA256, family/joint/span, path L/R, exact or outward interval coefficients, upper margin, decisions and reference quantities.
6. Implement a **separate independent verifier** (different code path for residual construction, dyadic partition and bound replay). It must not trust producer status flags; detect missing leaves, duplicate/overlapping leaves, forged bounds, altered coefficients, wrong family or source hash.
7. Validate against exact Fraction reference, MPFR directed rounding when available, and existing Python outward arithmetic; at least 10,000 seeded multi-joint coefficient cases per supported compiler/configuration, plus adversarial ULP/subnormal/signed-zero/NaN/Inf/overflow, high condition numbers, very small spans, strict zeros, near-active limits and proof budgets. Finite tests do not prove universal floating point correctness: document platform assumptions and conduct a manual arithmetic audit.
8. Profile separately parsing/extraction, input packing, polynomial construction, subdivision, validation and replay. Do not compare hull-only work directly against full certification.

**Gate P2:** every scope-appropriate exact-reference passing case accepted by producer+independent replay, every known violation/unknown respected, zero observed false certificates; new native-linkage requirement from P1 still separate.

## 6. P3 — construct a real hard-case corpus without changing TimeOpt

Before formal cohort registration, reuse the eight frozen Panda models with their original array hashes and add an exploratory stress set:

- Diverse obstacle-rich/near-collision queries, original physical limits, larger payload conditions if supported by source configuration, high-utilization velocity/acceleration/jerk cases, and low positive x.
- Seek truly feasible but initial-hull-inconclusive intervals: `U0=max(b)>0` with a defensible continuous peak upper `U*<=0`, or tight exact upper brackets. Distinguish actual violation witnesses, near-zero tangencies, strict positivity violations and unknown.
- Compute per-family peak bracket `L*<= max r <=U*`, original hull overestimate `U0-U*` brackets, physical margin, split depth/nodes and leaf sizes. Compare expected proof cost with observed runtime.
- Keep planning failures and invalid native outputs in the *submitted-query denominator*, but stratify native-success eligible cases for K3 targets. Distinguish physical vs solver-scaled limits.
- Export up to 250 selected residuals at a time for Proof Lab. The viewer's single-residual PASS must never appear as a whole-path or native pass.
- UR5e only after verified description source, joint order, kinematics, acceleration/jerk limits and collision model. Panda-only exploration may proceed earlier.

**Gate P3:** a frozen, hash-verified exploratory cohort including genuinely subdivision-demanding real native outputs, or explicit finding that they are rare with counts/selection strategy. No synthetic cases in the real-path denominator.

## 7. P4 — ablation matrix and cost decomposition (Dev40 only)

Once P1/P2/P3 correctness gates pass, freeze Dev40 by source scene/query (registered seed `20261110`), then run every method on the **identical** frozen outputs.

| Method | Meaning |
| --- | --- |
| H0 | Original Bernstein hull, no splits |
| U1/U2/U4/U6 | Uniform 1,2,4,6 subdivision levels, as registered |
| Adaptive | BFS de Casteljau adaptive proof |
| Adaptive-no-witness | Same budgets, without optional early witness handling |
| Sample S8193/S32768 | Falsification diagnostics ONLY; NEVER certification |

Sweeps: depths 0,2,4,8,12,16; node budgets 2048/8192/32768. Use a registered wall deadline only after complexity sweeps. Report status per constraint family, proof completeness, violation witnesses, unknown reason, leaves/depth, certificate bytes and exact hull gap.

Metric scopes:
- `hull_only_cpu`
- `frozen_output_full_certificate_cpu` (producer + **independent** verifier)
- `frozen_output_gpu_or_hybrid` (all transfers)
- `fresh_native_total` and **added** `fresh_native_total - fresh_native_without_cert`

Use 50 warmups; >=10 paired randomized AB/BA repeats per source path, same machine affinity/clocks/power/GPU and precision, separated CPU and GPU stages, raw repeat logs. p50/p90/p95/p99/max + source/scene-cluster 10,000-resample bootstrap 95% CIs (note p99 instability on small source counts). Record output duration and byte hashes before/after. RTX3090 primary; A100 replica only after frozen implementation.

**Gate P4:** honest coverage/latency trade-off with full denominators and no false certificate; identify reproducible bottlenecks; do not select a winner from mismatched workloads.

## 8. P5 — lock and qualify

After Dev40 tuning, lock code/config/binary/source hashes and run previously untouched Holdout100, then Confirm300. If code or certification acceptance changes after opening holdout, label affected cohort exploratory and use untouched samples for qualification. Report K3 >=99% eligible native-success, fresh added p50 <=1 ms / p95 <=3 ms separately from any full torque/torque-rate claim. Confidence interval grouping by scene/query, not individual repeated calls. Categorize decision PASS_BASELINE, CORRECT_BUT_SLOW, INCONCLUSIVE_DOMINATED, VIOLATION_DOMINATED, BLOCKED_NATIVE_LINKAGE or BLOCKED_DYNAMICS. Continue to show non-passing results honestly.

## 9. P6 — deepen interactive real-path diagnostics

V0 already has an exact single-residual teaching lab. The next development stages should add:
- Native trace bundle selector: trajectory → span → joint → constraint family; original vs serialized vs native sampled values and their status labels.
- A synchronized view of physical q(s), x(s), qdot, qddot and jerk with knot-side markers, original limits and proof bounding bands; never infer unsampled safety from a plot.
- Per-leaf initial and subdivided Bernstein control polygons and certificate trees, with clickable proof hash/interval/coefficients and independent replay.
- A linked hull tightness panel: physical margin, exact peak bracket, hull-gap and splits, highlighting where adaptive beats uniform.
- Pareto/ECDF plots of certified fraction versus complete certificate wall time, source-linked outliers, method comparisons and leaf count/latency correlation.
- Capture real screenshots and record offline browser test results for both Firefox and Chromium. All data bundles must embed model source SHA, scope, GPU/source revision and explicit NOT_EVALUATED fields. Never silently replace missing real results with synthetic numbers.

## 10. Required proof and performance reports

Persist `environment.json`, `source_models.jsonl`, `native_operand_traces.jsonl`, `certificates/<path-id>.json`, `requests.jsonl`, `summary.json`, and an offline `reports/...` HTML with source SHA and proof-level legend. Keep raw captured paths in ignored `runs/` if sensitive/large; commit scripts, summary tables, safe small fixtures and documentation only.

Each milestone: inspect -> implement -> unit and adversarial tests -> diagnose bugs -> rerun -> benchmark -> commit -> append `docs/PROGRESS.md` with exact commands/hardware/versions/denominators/results and blockers. Never silently substitute synthetic data for real Panda, never count exact-polynomial as native FP32 certificate, and never call partial C++ hull-only output a fully replayed K3 proof.

## 11. Immediate next action

First reproduce and visually check V0 Proof Lab offline, test JSON export from the analytic fixture, and reconcile the reported local `1100caaa` integration. Then **parallelize P1 (native arithmetic/linkage) and P2 (compiled full proof + independent replay)** with separate commits and disjoint file ownership. P3 hard-case harvesting can run alongside both, but formal Dev40 must wait for gates. No PR or merge.
