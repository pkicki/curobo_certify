# K4 P5 independent review: compiled K4 backend and P6 acceptance logic

- **Date:** 2026-11-17.
- **HEAD reviewed:** `922eaa6a8cc8a7fa5b0399034550f8dce8232d29`.
- **Reviewer:** the same isolated Claude subagent as P1/P2. It did not write the reviewed code, made no commits, used no GPU and modified no existing file.
- **Created:**
  - this memo;
  - `compiled_vs_oracle.json`;
  - `tests/test_independent_k4_p5.py`.

## Verdict

**Compiled backend (`native/k4_rnea.cpp` via `k4_compiled`): PASS for soundness. One low hygiene finding, F-P5-1.**

The compiled interval and dual-interval RNEA encloses the true A, B and G, and their s-derivatives, in every comparison against the independent rigorous Lagrangian oracle (`independent_checks/k4p2`):

| Check | Result |
|---|---|
| Exact states, rigorous iv oracle (25 states) | 875 / 875 contained |
| Random sub-boxes, 160-bit dense samples (25 boxes) | 7,000 / 7,000 contained |
| tm1 slope boxes on random leaves vs 160-bit central-difference derivatives | 1,008 / 1,008 contained |
| Isolated compiled sin/cos, exact range (308 intervals including ±8, nextafter(8), π/2, subnormal) | all contained |
| Compiled tm1/box GeometryCache on the real fixture (physical, 0.34 and 0.3 limits) | oracle-contained leaves; `verify_k4` passes; witnesses confirmed |

Its use in P4 is gated by the mandatory Python world-frame `verify_k4`, so a compiled-backend error could not by itself produce a reported CERTIFIED.

**P6 acceptance (`scripts/k4_retime_run.py`, `k4_retime.py`): PASS.**

- No candidate is accepted unless both of these hold for the exact float32 controls that are reported:
  - MODEL_K3: compiled producer CERTIFIED and the independent verifier passing;
  - MODEL_K4: tm1 CERTIFIED and `verify_k4` passing.
- Reported durations are the rigorous time-law brackets (`time_law_grid`, `BOUNDED_EXACT_MODEL`, 2^16 cells) of those controls. The gain is computed conservatively as (T_native_lo − T_retimed_hi) / T_native_hi.
- The optimiser itself is heuristic and is correctly given no proof role.

There is one interpretation note, N-P6-1, below.

## Obligation matrix

Line references are to `native/k4_rnea.cpp` (K) and `k4_compiled.py` (KC) at 922eaa6.

| # | Obligation | Verdict | Evidence |
|---|---|---|---|
| P5-1 | Outward +, −, ×, sqr | PASS | K:30-48. Each op is round-to-nearest, then widened one ulp with `std::nextafter`. In `operator*`, any NaN product (0·∞) gives [−∞, ∞]. Same algebra as the reviewed Python IA (P2-1/2). |
| P5-2 | Interval Taylor coefficients via `idiv_pos` | PASS | K:41-65. 1/(2k+e)! comes from repeated correctly rounded division by exact integers n ≤ 61, each widened outward; the sign is applied exactly. Covered indirectly by the trig test (`test_compiled_sin_cos_enclose_exact_range`). |
| P5-3 | REM = 1e-27 for \|mid\| ≤ 8 | PASS | Exact rationals: 8^62/62! < 1e-27 (sine, degree 61) and 8^61/61! ≈ 2.4e-29 < 1e-27 (cosine, degree 60). The bound applies because K:69 falls back to [−1,1] when `!(fabs(m) <= 8)`, which includes NaN (`test_remainder_constant_covers_lagrange_bound`). |
| P5-4 | Lipschitz radius, clamp, fallback | PASS | K:67-78. The radius is computed outward around the float midpoint, which covers subnormal or inexact midpoints. 308 intervals were checked against the exact range; point widths are at most 3.6e-12 at \|x\| ≤ 8. |
| P5-5 | Dual intervals | PASS | K:80-93: product rule, plus sin′ = cos·d and cos′ = −sin·d over the value box. Slopes contain the true dA/ds, dB/ds and dG/ds: 1,008 / 1,008 (`compiled_vs_oracle.json`, `test_tm1_slopes_contain_true_derivative`). |
| P5-6 | Templated RNEA equals the reviewed link-frame RNEA | PASS | K:110-188 mirror `k4_panda_model.rnea` line for line (base acceleration −g; R = Rf·Rz(q); rotations and translations; CoM inertia; backward pass). G: v = a = null. B: a = q′, then ×0.5 exactly. A: v = q′, a = q″. Oracle containment as in the Verdict table. |
| P5-7 | Build flags / no contraction / no fast-math | PASS (F-P5-1) | KC:22 sets `-O3 -std=c++17 -fno-fast-math -ffp-contract=off`. K:15-17 refuses `__FAST_MATH__` at compile time (`-ffast-math`, `-Ofast`). Even if FMA contraction occurred, the one-rounding result would stay inside the widened enclosure. Compiler: g++ 9.4.0. **F-P5-1 (low, hygiene):** `build(extra=...)` accepts individually unsafe flags that do not define `__FAST_MATH__`: `-ffinite-math-only` (can remove the `isnan` guard in `operator*`), `-fassociative-math`, and a later `-ffp-contract=fast` (`test_FINDING_build_should_refuse_individual_unsafe_flags`, expectedFailure). Current callers (`k4_bench_fresh.py`, tests) pass no extra flags. Recommend an allowlist. |
| P5-8 | Parameter packing and ctypes layout | PASS | KC:35-44 packs the exact PandaModel interval enclosures in order R(9), p(3), m, c(3), Ic(9), which is 25 intervals or 50 doubles per link, then g. This matches K:190-206. Input blocks are q, q′, q″ (and q‴) per box, as lo/hi pairs; output is A, B, G. Arrays are C-contiguous float64 and int32 (`c_int`). Agreement with the Python reference is tested in `tests/test_k4_compiled.py` and against the oracle here. |
| P5-9 | Non-finite inputs | PASS (same as Python) | A NaN or ∞ angle gives sin, cos ∈ [−1,1], identical to the Python reference and sound for any real angle. Exact-rational path boxes never contain NaN. A NaN velocity or acceleration propagates to a non-finite enclosure, which is rejected (`test_nonfinite_inputs_fail_closed`). |
| P5-10 | Input validation in C | NOTE | `load_model` checks only n ≤ 32 and dof ≤ 16. Parent and joint indices are trusted, and are validated upstream by `PandaModel` (topological order, joint coverage). A direct ctypes caller with a bad topology could read or write out of bounds. Low, with no current exposure. |
| P5-11 | Thread safety | NOTE | `init_coeff` uses an unsynchronised global flag. ctypes releases the GIL, so concurrent first calls race, although they write identical values. Low; recommend a static initialiser. |
| P5-12 | `backend=` path in `GeometryCache.ensure` | PASS | With `backend=None` the behaviour is identical to the reviewed Python path. With a backend, A0/B0/G0 come from `backend.coefficients` on the centre point box, and A1/B1/G1 from `backend.slopes` on the leaf boxes of q..q‴. That is the same Taylor-model construction (P2 addendum). Compiled tm1 and box caches pass oracle containment, `verify_k4` and witness replay on the fixture (`test_compiled_tm1_cache_leaves_contain_oracle_and_verify`). |
| P5-13 | P4 usage gate | PASS | In `scripts/k4_bench_fresh.py`, a compiled-backend K4 CERTIFIED is reported as CERTIFIED only after `verify_k4(..., refine=8)` passes; otherwise it becomes REJECTED_BY_VERIFIER. |
| P6-1 | No uncertified candidate accepted | PASS | `accepted` is assigned only under `st == ("CERTIFIED", "CERTIFIED")`. K3 is downgraded unless `V.verify` passes; K4 is downgraded unless `verify_k4` passes. The accepted array is the exact float32 `cand` that was proven. An AST check and a scripted simulation of the bisection loop with adversarial proof outcomes are in `RetimeAcceptance`. |
| P6-2 | Durations are rigorous brackets | PASS | `duration()` returns `g.duration` only for `BOUNDED_EXACT_MODEL`, otherwise None, in which case no gain is reported. The bracket is computed on the certified float32 controls. Gain = (Tn_lo − Ta_hi)/Tn_hi, which is conservative. |
| P6-3 | Optimiser role | PASS | `sampled_constraints` and `slp_optimize` are labelled heuristic: sampled midpoints, 0.98 safety factor, linearised jerk. Nothing from them reaches the report except through the K3+K4 proofs. |
| N-P6-1 | Comparison semantics | NOTE (scope) | Candidates are certified against the PHYSICAL limits: effort 87/12 and physical K3 limits. Native TimeOpt plans at 0.95 of those. Part of a reported "gain vs native 0.95" may therefore be consumed safety margin rather than better timing. Report the gain alongside the margin used, or also certify retimed candidates at 0.95. The docstring mentions "paired native 0.99", which this script does not implement. |

Not reviewed: the compiled K3 producer and verifier (`compiled_k3`), which were reviewed in earlier K3 work, and the GPU path of the bench.

## Tests

- `tests/test_independent_k4_p5.py`: 12 tests, OK, 1 expected failure (F-P5-1), about 30 s. It builds the library in a temp dir and skips if g++ is missing.
- Full suite at 922eaa6 (working tree plus reviewer files): **380 tests OK (skipped=6, expected failures=1)**, 225 s.

## SHA-256 (at 922eaa6)

```
99c0e9a1a1dfc64f9b90e830ab8e8cf6e62eff1d980b845d631298d5d4fcb6cd  src/curobo_certify/native/k4_rnea.cpp
5fae762e2980e2b7fe739de91b50579600c5aaa03c0ac0ed50249ba4f35da1fa  src/curobo_certify/k4_compiled.py
0430158ceef03ac8aa598dbfa473970d11722427266222104b35ee5d8f3dd22a  tests/test_k4_compiled.py
c5b80ee110d917a81c8c8e7ddd008e5860e4e12f5fae74af906108a8df43618e  src/curobo_certify/k4_reuse.py
c5a3ca9f727f535a6e376d43251bbac4e786697ccd01e7f0601755d60a626785  scripts/k4_retime_run.py
f4ca34e35779c7304637381a603f5f91547c977b0d235763517ec204fb4c94e0  src/curobo_certify/k4_retime.py
da2c641b7dd5a516f7833cc9b0333b528b74fe5fd0ec062d933ad80788b988d7  scripts/k4_bench_fresh.py
8456e19076f614c3e388cf824d0ee761b0697e34ab93a0057adc7df61464e5eb  tests/test_independent_k4_p5.py (reviewer)
898742d0103acd28702208604d3ff534f1833ce483e96bdb02bd93f8b1a924b1  docs/reports/k4_p5_independent_review/compiled_vs_oracle.json (reviewer)
```
