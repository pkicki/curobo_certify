# Coding-agent handoff — finish margin-duration-native-time research

You are working on pkicki/curobo_certify. New branch exp/margin-duration-native-time-20261117 descends from real integrated certification result 5c7c0fcf. Read docs/MARGIN_DURATION_NATIVE_TIME_CAMPAIGN_20261117.md COMPLETELY, then docs/PROGRESS.md, docs/NATIVE_LINKAGE_20261116.md, docs/COMPILED_K3_CERTIFIER_20261116.md and all new scripts/tests. Follow the detailed registered experiment protocol rather than improvising an easier one.

USER EXPECTATION: You implement, test, debug, benchmark, commit and report; do not ask for routine approvals, do not create PRs, do not merge to main, do not force-push. Preserve original frozen NPZs and ignored experimental data, original actuator limits, planner path/timing output and source provenance. Work on this experiment branch or a new child branch after inspecting Git state. Commit logical passing packages and push them to origin.

IMPORTANT STATUS: The preparing assistant used the GitHub connector to commit a substantial initial implementation WITHOUT running the full Python tests, browser tests or any new GPU experiment. DO NOT TRUST IT until independently validated. The offline dashboard contains no fabricated data. All existing 79-path results are previous exploratory data, not fresh validation of new code.

Research objectives:
1. Matched Panda planning-scale sweep (0.90/0.95/0.975/0.99/1.00) with 36 frozen source queries × five margins = 180 attempted cells; 4 scenes and three motion magnitudes, 3 source queries per stratum. A single pair_id has identical scene/start/goal across five arms.
2. Real trajectory-duration vs continuous-time certifiability tradeoff, accounting for all failures and missing proofs; never confuse certified exact-model splines, conditionally certified native-reported samples and derivative correctness of a continuous physical-time command.
3. Rigorous native time-inversion error contract, including strict-rest endpoints and actual floating-point kernel semantics; failure is an acceptable research outcome.
4. Reduce measured ~9.7 ms compiled native-bound stage and ~13.1 ms fresh added overhead without weakening proof conditions; independently verify every produced certificate and benchmark only same-frozen-output workloads.
5. Produce interactive offline real-results report with case-linked Bernstein hulls/witnesses/physical limits and paired duration statistics.

Immediate action list:
A. Inspect git branch/HEAD/status/ignored files, pinned cuRobo SHA c301cb00a87012c2808b1a60649c4065c9b3c95a, previous 79-path reports, Python and compiler environment. Preserve all historical source data.
B. Run and FIX, with regression tests: Python unittest discovery, compileall src/scripts/tests, Node test suite, built standalone old/new HTML, full Chromium and Firefox desktop/mobile smoke. Fix inherited CI Python 3.13 mismatch with pyproject's <3.13 requirement and include new branch in trigger.
C. Audit every new code path. Particular potential pitfalls: runtime-only margin override reaching the *actual* TimeOpt config; default 0.95 retains unpatched constructor; old generator summary by_condition hardcodes only two arms; frozen query matching and SHA associations; planning contract in bench_certification_methods.py needs correct per-arm scale; benchmark statuses for tightened-limit witnesses must be UNKNOWN; no invalid zero-speed endpoint acceptance in exact time-integral prototype; dashboard should not miscompute source denominators or silently show missing results; native table cache should be genuinely immutable and parity-preserving. Fix all defects, document before/after test evidence and commit.
D. Freeze matched queries BEFORE solving:
  python scripts/freeze_margin_queries.py --seed 20261117 --per-cell 3 --out runs/margins/queries.json
Verify 180 rows, 36 complete matched pairs, manifest SHA, original physical limits, and two scale properties on actual instantiated solver. Create a reproducible environment manifest.
E. GPU pilot:
  python scripts/generate_hardcase_population.py --queries runs/margins/queries.json --out-dir runs/margins/native
Run on RTX3090, isolated/registered state and order; preserve all 180 planned outcomes, failed attempts and output SHA/duration. No post-hoc filtered restarts or output modifications. Diagnose nondeterminism separately.
F. Complete frozen output certification, using source-backed planning scales for model/physical/planning/native-reported contracts, producer+independent verifier, H0/U1/U2/U4/U6/adaptive/no-witness. Compare identical actual output NPZs and whole-pipeline wall time. Do not call a witness vs tightened native bound a real physical violation. Distinguish source/ISA conditional native-reported values from real time derivatives.
G. Generate paired statistical analysis with source-pair bootstrap CIs, full denominator/success/missingness and per-constraint exact violation/unknown/leaf count; reject source SHA conflicts. Commands and scope in the full plan. Generate offline interactive margin dashboard and update genuine real-path Proof Lab/report sections.
H. Independently investigate native time-law bridge: exact-positive-speed Fraction integrals now exist but intentionally UNKNOWN at zero-speed endpoints. Build mathematically sound root-multiplicity endpoint substitutions and an independent verifier; audit native table / lookup / 511-cell Newton algorithm, PTX/SASS, FP32/FMA/FTZ/div/sqrt assumptions. Prove a meaningful enclosure of s_native(t)-s_exact(t), not impossible bitwise equality at every real time; define continuous commanded trajectory interpolation and physical-time derivative contract or leave BLOCKED_NATIVE_TIME_INVERSION. Distinguish queried time vs continuous time. Never relax registered 4e-4 parity threshold.
I. Optimize compiled native bound: first profile cached timing table and shared timing stage, then conservative per-span/joint precomputation, SIMD/CPU, and only consider GPU if justified. Run scripts/bench_native_bound_pieces.py for original frozen NPZ with pieces=4/8/16/32, div_ulps=2/4, 50 warmups >=10 paired repeats. Every configuration must retain independent verifier and soundness assumptions. p50/p95/p99, source-cluster CI, decision status, bound enclosure, proof leaf status, input hashes. If wrong output or suspected underenclosure, STOP, debug, rerun.
J. Formal Dev40/Holdout100/Confirm300 stay NOT RUN until native physical-time/ISA correctness gates actually pass and code is frozen. Do not claim qualification on only exploratory results.

Research status and paper: Prior 79 real trajectories showed 12 exact mathematical violations in 31 margin=1.00 outputs, 19 certified and 48/48 default margin=0.95 mathematical feasible. Test if increased planner margins reduce duration but worsen feasibility or proof cost; no guaranteed positive result. Predeclare interpretation, avoid selecting best margin from same test data as claimed policy. Existing reported added p50 ~13.1 ms falls far short of <=1ms engineering target; don't present plans as measurements.

Requirements for final response:
- Branch and final commit SHAs pushed; main unchanged.
- Exact environment source, git/driver/CUDA/NVRTC version pins and source data SHA.
- Number/passes/failures of Python, Node, browser and mathematical differential tests, commands and fixes.
- Per-margin submitted/native-success/failed/model-K3-certified/math-violated/native-reported-certified/physical-time-certified/UNKNOWN/NOT_RUN counts, plus duration and paired source-cluster 95% CI; clear denominators.
- Unchanged original path/timing/duration/hash checks; source-query match.
- Inter-sample violation witnesses with true exact mathematical bounds, and any native arithmetic caveats.
- Native inversion progress bounds or precise BLOCKED reason.
- Complete frozen and fresh latency breakdown with A/B CI, including native bound producer and verifier, with no hull-only vs full-proof comparison.
- Offline HTML/report path with real interactive examples and no synthetic claims; exact screenshots from desktop/mobile browser acceptance.
- Explicit GO/NO-GO for paper, for native reported-value deployment, and for native physical-time certification. Keep qualifying cohorts closed if the correctness gate fails.

Continue autonomously, investigate and fix bugs; negative results and honest blockers count as valid outcomes.
