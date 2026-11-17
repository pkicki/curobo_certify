# K4 P2 re-review addendum: fixes for the P2 findings

**Reviewer:** the same isolated Claude subagent as P1/P2. No commits, no GPU. Only this file and
`tests/test_independent_k4_p2.py` (reviewer-owned) were changed.

**Commits:**
- Fixes reviewed at **`c5bd01e38ec33b566f80246ac8bbf6ce2f5c2da8`**.
- HEAD then advanced to **`2f4427de99db474818327034c26b2dd76b1d2b0e`** (P5 compiled backend).
- Between them, the only reviewed-file change is in `k4_reuse.py`: an optional `backend=` parameter.
  With the default `backend=None` the behaviour is identical. The compiled C++ backend
  (`k4_compiled.py`, `native/k4_rnea.cpp`) is NOT reviewed.

## Final P2 gate verdict

**Soundness: PASS.** **Gate: CONDITIONAL PASS. One blocking item: R1.**

- **Soundness.**
  - Every soundness obligation from the P2 memo still holds.
  - The re-run oracle tests show 0 counterexamples. This covers:
    - producer, verifier and GeometryCache box/tm1 against the rigorous Lagrangian oracle;
    - fixture certificates with physical and 0.34-scaled limits, and with an alternative timing;
    - witnesses at 0.3-scaled limits, with the oracle confirming every witness.
- **The fixes.**
  - Fixed: F-P2-1, F-P2-2, F-P2-3, F-P2-5, F-G3, N-IA1 (for ints), N-C1, N-V1.
  - Partially fixed: F-P2-4.
- **R1 (new regression, introduced by the F-G3 fix).** It is fail-closed, not unsound.
  - `k4_affine.certify_with_prepared_geometry` never sets `input_digest`.
  - Because an empty digest is no longer a bypass, every genuine affine CERTIFIED now fails
    `replay_torque_leaves`.
  - Three tests fail at HEAD:
    - `tests/test_k4_affine.py::GeometryReuseTest::test_reuse_fixed_path_vary_timing` (the
      coordinator's own test);
    - `tests/test_independent_k4_oracle.py::AffineAttacks::test_signed_xprime_alternative_timings`;
    - `tests/test_independent_k4_oracle.py::PathCertificationAdversarial::test_randomized_rational_families_no_counterexample`
      (`affine_replay_rejected > 0`).
  - The coordinator's message listed 5 unexpected successes and 3 FAILs, all in
    `test_independent_k4_p2.py`. It missed these 3 failures.
  - **Fix:** in `certify_with_prepared_geometry`, return
    `dataclasses.replace(result, input_digest=input_digest(spans, model, limits))` (on every
    status). Keep the strict replay.
  - Once R1 is fixed and the suite is green, P2 = PASS.

## Per-finding verdicts

| ID | Fix at c5bd01e | Verdict | Evidence / residual |
|---|---|---|---|
| F-P2-1 contract hash | `k4_verify._contract_ok`: recomputed canonical hash == recorded == `FROZEN_CONTRACT_V2_SHA256`; used by `verify_k4` and `verify_k4_witness` | **FIXED** | Each of these is rejected by both the verifier and the witness check: tampered mass, zero gravity, and a tampered contract re-hashed self-consistently (it is not the frozen SHA). Test: `test_verifier_checks_contract_hash`. Future contracts need `expected_contract_sha256=` passed explicitly, which is intended. |
| F-P2-2 [0,1] tiling | `spans_tile_unit_interval`: left₀ = 0, contiguous, right_last = 1, positive lengths. Used in `certify_path`, `verify_k4` and `GeometryCache.certify` | **FIXED** | Rejected: a prefix, a missing first span, an interior gap, reversed order, and two swapped spans. Test: `test_span_coverage_of_unit_interval`. |
| F-P2-3 input binding | `k4_input_digest`: contract SHA, limits (float hex), exact span q and x coefficients. Set by the producer and GeometryCache; required by both verifiers | **FIXED** | Rejected: empty digest, wrong digest, and an altered timing on one span (`test_path_result_binds_inputs`). The digest does not cover leaves, which is correct since the verifier re-proves them, nor the code version (note). |
| F-P2-4 recorded boxes | The verifier now requires recorded boxes to be finite and inside the limits | **PARTIAL (low, open)** | A box outside the limits is rejected (`test_recorded_leaf_boxes_outside_limits_rejected`). Zeroed, under-enclosing boxes and utilisation still pass. Open test: `test_FINDING_under_enclosing_recorded_boxes_should_be_rejected` (expectedFailure). Either check recorded ⊇ verifier box, or label the recorded boxes informational. |
| F-P2-5 non-finite limits | Finite limits are required by the producer, GeometryCache and verifier | **FIXED** | `test_nonfinite_limits_rejected` |
| F-G3 empty digest | Both replays require an exact digest match | **FIXED, causes R1** | `test_stripped_digest_does_not_bypass_binding`; R1 above, `test_FINDING_R1_affine_certificates_should_replay` (expectedFailure). |
| N-IA1 IA.const/of/scale | Python ints are enclosed outward; `scale()` rejects inexact Fraction/int | **FIXED (residual low)** | `test_const_of_int_encloses_and_scale_rejects_inexact`. Residual: numpy integers, str and Decimal still round to nearest via `float()` (`test_residual_nonint_nonfraction_inputs_are_rounded`). No call site is affected. Recommend `TypeError` for anything other than float/int/Fraction. |
| N-IA2 FTZ check only at import | Not addressed | OPEN (low) | Unchanged. |
| N-C1 prismatic lock with a rotated origin | The contract script now rejects it | **FIXED** | Diff to `scripts/k4_model_contract_v2.py:72-74`. The contract itself is unchanged (SHA `851459c3…`, model SHA `9caa38bf…`). |
| N-V1 malformed input raised | `verify_k4` catches errors and returns `pass: False` | **FIXED** | `test_verifier_bad_path_symbol_rejected` covers a bad symbol, a `None` path, and a non-result object. |

## The 3 tests the coordinator changed on purpose

I agree all three are intended behaviour changes. None of them hides a regression.

1. **`test_const_of_nonfraction_hazard_is_documented`.** Ints are now enclosed. The test was
   rewritten to assert enclosure, plus `scale()` rejection, plus the documented residual.
2. **`test_verifier_bad_path_symbol_raises_instead_of_rejecting`.** It now returns a fail; the test
   was rewritten accordingly.
3. **`test_limits_physical_gate`.**
   - Binding a result to the limits it was produced with is stricter and still sound. Previously,
     re-verifying under other limits was a fresh re-proof, so nothing that was valid is lost: a
     scaled-limit certificate must be produced with the scaled limits.
   - Rewritten to cover these cases:
     - **Rejected:** the physical result checked against 0.95-scaled limits.
     - **Accepted:** the 0.95 result, which is CERTIFIED, verifies with `expect_physical=False`.
     - **Rejected:** that 0.95 result checked as physical, under either set of limits.

## Short soundness look: `k4_interval.DIA` and `k4_reuse` (tm1, box)

This is not a full P3 review.

- **DIA** (`k4_interval.py`):
  - The forward-mode dual interval is v + d·ε, with the product rule `d = d₁v₂ + v₁d₂`.
  - sin and cos carry derivatives `cos(v)·d` and `−sin(v)·d`. Both `_trig` evaluations run over
    the whole value box, which is a valid chain-rule enclosure.
  - IA operations return `NotImplemented` when the other operand is a dual, so mixed expressions
    dispatch to DIA's reflected methods. `IA − DIA` reaches `DIA.__rsub__`.
  - The RNEA code path uses only + − × `scale` sin cos, with no branches on values. Sound.
- **tm1** (`k4_reuse.py:95-119, 179-192`):
  - Taylor model: C(s) ∈ C(s_c) + [C′]·(s − s_c), where [C′] encloses dC/ds over the whole leaf.
    It is computed by DIA with dq/ds = q′, dq′/ds = q″, dq″/ds = q‴, all global-s derivatives
    via `dpoly(…, h)`. By the mean value theorem this is valid per joint.
  - The main part G0 + A0·x(s) + B0·x′(s), with constant interval coefficients, is bounded by the
    hull of Bernstein coefficients via the partition of unity. The remainder
    (G1 + A1·X + B1·X′)·[−H, H] uses the leaf hulls of x and x′, with H widened up.
  - Witnesses use the uncached rigorous point RNEA. Sound.
  - **Evidence:** box and tm1 caches on the fixture, at physical and 0.34-scaled limits, with an
    alternative timing and with 0.3-scaled (violated) limits:
    - every CERTIFIED leaf is CONTAINED by the rigorous oracle and passes `verify_k4`;
    - every witness is confirmed (`GeometryCacheAgainstOracle.test_box_and_tm1_leaves_contain_oracle_and_verify`);
    - earlier ad-hoc run: box and tm1, 112/112 CONTAINED, 2/2 witnesses confirmed.
- **L-TM1 (latent, low–medium for P4 use).**
  - tm1 elevates x and x′ to degree 3 and reads exactly 4 Bernstein coefficients. A span with
    x of degree above 3 would silently drop coefficients.
  - In the cases I built, the B′x′ remainder happened to cover the drop, so I have no
    counterexample. Coverage is not guaranteed in general.
  - `extract_timeopt_spans` always yields cubic x, but `GeometryCache.certify` accepts any `Span`.
  - Recommend `INVALID_INPUT` when `len(sp.x) > 4` (and q degree > 3) in both GeometryCache modes.
    Open test: `test_FINDING_tm1_should_reject_x_degree_above_3` (expectedFailure).
- **Before P4 relies on it:**
  - Every tm1/box CERTIFIED is still checked by the independent world-frame `verify_k4` (full
    recomputation from contract v2). The P4 protocol should keep that verification mandatory.
  - The compiled backend (`backend=`) needs its own review before it becomes the producer.

## Results at the 2f4427d working tree (plus this reviewer's updated test file)

- `tests.test_independent_k4_p2`: 35 tests OK, 3 expected failures (F-P2-4 partial, R1, L-TM1), about 72 s.
- Full suite: **368 tests, FAILED (failures=3: R1), skipped=6, expected failures=3**, 219 s.

## SHA-256

At 2f4427d. Files marked (*) are unchanged since c5bd01e.

```
f1a815857e663f52a7aae5c7b9e73c40ea32149a77ac5a833109aa869c984fb9  src/curobo_certify/k4_interval.py (*)
ba6db49568e0736ae1ed2f1eb601c6666582791344094e5795ba87c61b946bf1  src/curobo_certify/k4_path.py (*)
06b622ba625b732ef186bb0522bd4b8cca513728e7463b6aff3a2fdc9659e523  src/curobo_certify/k4_verify.py (*)
d6711bd0defd30959add95edd6c069b689e0ee6229d94460668e38146dee910c  src/curobo_certify/k4_reference.py (*)
dd63358e49de6f2fd7eb2f2b6267b52a765fe89ec062878fbc180a94ecb0c6d3  src/curobo_certify/k4_reuse.py  (c5bd01e: ab3dd5f0d258…; diff = optional backend only)
a3448cd6f2fa7fe4d96b8d4ad0a21be95821f20db03ffd4ee4b1918c85f309f3  src/curobo_certify/k4_affine.py (unchanged since P1; source of R1)
29c274114f795eec6b5bc7bb957d8e92fcb424225fec89807af40b091473c46f  src/curobo_certify/k4_panda_model.py (*)
22c132bb7a1921a6394e2fbba699825ca5be7164cb42dceaa9c74b594eae7fb9  scripts/k4_model_contract_v2.py (*)
851459c341974b7be32a3bb6ff93bb2d6f9561493d31833de71c3e0a4c5e6b36  docs/reports/k4/p0/panda_model_contract_v2.json (*)
54da889d7d783889fdcd095d30b25602d49f6dbfa4d20b32a2569a1b0ce9e46e  tests/test_k4_path.py (*)
a7ffced992404703847c82c7b7e616da06e9808b30004521537debed184d61a1  tests/test_independent_k4_p2.py (reviewer, updated)
```

---

## Final confirmation at `d65ff61` (re-checked at HEAD `4962bbef710bb239bfa9b1d101fd83d4f5187ab7`)

`4962bbe` only adds P6 retiming code (`k4_retime.py`), which is outside P2 scope and was not reviewed. All
P2-reviewed files are identical between `d65ff61` and `4962bbe`.

| Item | Fix | Verdict |
|---|---|---|
| R1 affine replay regression | `certify_with_prepared_geometry` wraps the result and sets `input_digest` on every status | **FIXED**. A genuine affine certificate replays, and the replay is bound to its limits. The three tests broken by R1 pass. |
| L-TM1 degree > 3 in tm1 | `GeometryCache.certify` returns INVALID_INPUT when q or x is above cubic | **FIXED** in both modes, for both quartic x and quartic q (`test_tm1_rejects_degree_above_3`). |
| F-P2-4 recorded boxes | The rigorous world-frame point torque at each leaf midpoint must lie inside the recorded box (1e-9 relative slack); foreign span indices are rejected before indexing | **FIXED as falsification.** Zeroed boxes, boxes shrunk to their central 0.1%, and a foreign span index are all rejected. Residual (low, accepted): only midpoints are checked, so a box deliberately built around the true midpoint torque but too narrow elsewhere would pass. The CERTIFIED verdict never relies on recorded boxes: the verifier re-proves every leaf with its own enclosures. Treat recorded boxes and utilisation as checked-at-midpoint diagnostics. |

**Test conflict** (`test_verifier_partition_attacks`): I agree that zeroed recorded boxes are false claims and must be rejected. The test now checks two cases:
- **Re-partition with honest boxes, PASS required:** span 0 is split into L/R, both carrying the parent leaf's producer enclosure, which contains both children.
- **Zeroed boxes, REJECT required:** the same re-partition with zeroed boxes, plus a foreign span index of 10^6.

**Tests at the 4962bbe working tree, with the reviewer's updated test file:**
- `tests.test_independent_k4_p2`: 35 OK, 0 expected failures. Every finding is now a plain regression assertion.
- Full suite `PYTHONPATH=src:. python -m unittest discover -s tests -p 'test_*.py'`: **368 tests OK (skipped=6)**, 211 s.

SHA-256 at 4962bbe, for the files changed since the previous addendum section:
```
95a3836638d31d77b6cf1c46574e14c64c7ac0e76343238d461a4a0ef2fce190  src/curobo_certify/k4_affine.py
c5b80ee110d917a81c8c8e7ddd008e5860e4e12f5fae74af906108a8df43618e  src/curobo_certify/k4_reuse.py
8796dce1bdbc995f7ed1d29f094f30aed82572d35799ddc3198c28b7968e852d  src/curobo_certify/k4_verify.py
f7d5587bde1b1f574a63c12ddf3adc2ad455ca648d3d7f598c3551fa005a02f9  tests/test_independent_k4_p2.py (reviewer)
```

### FINAL P2 GATE VERDICT: **PASS**

This covers the analytic K4 reference, the Panda interval-RNEA producer (`k4_path`), the world-frame verifier (`k4_verify`), contract v2, and the default (Python) box/tm1 GeometryCache.

- **Soundness:** no unsound CERTIFIED, no false witness and no under-enclosure was found against the independent rigorous Lagrangian oracle.
- **Findings:** all are fixed and regression-tested.

Remaining low or open items, none blocking:
- N-IA2: the FTZ check runs only at import.
- Residual IA input types (numpy integers, str, Decimal) are rounded rather than enclosed; no call site is affected.
- The midpoint-only recorded-box check described above.

Scope:
- `CUROBO_PINNED_PANDA_RIGID_BODY_MODEL`; physical Panda fidelity remains BLOCKED_MODEL_PHYSICAL_VALIDITY.
- The compiled C++ backend (`k4_compiled`, `native/k4_rnea.cpp`) is NOT covered and needs its own review before it becomes a producer.
- P4 should keep independent `verify_k4` mandatory on every certificate.
