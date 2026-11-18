# K4 P5b independent review: compiled K4 pipeline (C++ producer and C++ verifier)

Date: 2026-11-18.

- **HEAD reviewed:** `a913a13eeaac00b7e4adb9041642a35f5c3d71f3`.
- **Reviewer:** the same isolated Claude subagent as P1/P2/P5. It did not write the reviewed code, made no commits, used no GPU and modified no existing file.
- **Created:**
  - this memo;
  - `campaign.json` and `run_campaign.py` in this directory;
  - `tests/test_independent_k4_p5b.py`.
- **Disclosure:** the coordinator edited my P5 test file in `dfbc364`. It converted F-P5-1 to an assertion and gave the P6 simulation env the new `full_certify` arguments. Both changes are disclosed in that commit and match the P5 findings.

## Verdict

**PASS. No false CERTIFIED and no false witness was found, and the C++ verifier is sound and substantially independent.** Four low or doc findings are listed below; none affects soundness.

**Evidence against the independent Lagrangian oracle.** The oracle evaluates the EXACT spans of `spline_polynomials.extract_timeopt_spans`, a third path construction that was reviewed in K3. Rigorous mpmath.iv was used on the spans.

**Real fixture, at four limit factors.**

| Factor | Status | Oracle check | Verifier |
|---|---|---|---|
| 1.0 (physical) | CERTIFIED | leaves contained | PASS |
| 0.34 | CERTIFIED | leaves contained | PASS |
| 0.3350799560546875 (P2 boundary, certified side) | CERTIFIED | leaves contained | PASS |
| 0.33507656921997064 (violated side) | VIOLATED_TORQUE | witness confirmed | — |
| 0.3 | VIOLATED_TORQUE | witness confirmed | — |

- The status matches the Python tm1 producer at every factor, with identical leaf sets at physical limits (coordinator's test).
- The oracle's true joint-4 peak lies between the two boundary limits: 29.151662 < 29.151767 < 29.151956 N·m.

**Random campaign** (`campaign.json`): 40 random float32 Panda paths × 3 limit scales.
- 120 runs: 40 CERTIFIED, 75 VIOLATED_TORQUE, 5 VIOLATED_X.
- 1,120 / 1,120 CERTIFIED-leaf oracle points contained.
- 75 / 75 torque witnesses and 5 / 5 x witnesses confirmed.
- The C++ verifier passed 40 / 40 honest certificates.
- 0 / 75 forged certificates were accepted. Each forgery was built for a truly violated path: depth-0 leaves covering every span, recorded boxes equal to the limits (so they contain the midpoint torque and lie inside the limits), and a correctly recomputed digest. They were rejected with refine = 12.
- 0 / 120 mismatches between 1 and 8 threads.

## Obligation matrix

Line references: `native/k4_rnea.cpp` (P), `native/k4_verify.cpp` (V), `k4_compiled.py` (KC).

| # | Obligation | Verdict | Evidence |
|---|---|---|---|
| B1 | Bit-level `up`/`dn` | PASS | P:23-30 and V:24-30. NaN and +∞ are returned unchanged. ±0 goes to the smallest subnormal. Positive values increment the bit pattern and negative values decrement it, which matches `nextafter` toward +∞, including −∞ → −DBL_MAX and −min_subnormal → −0. `dn(x) = −up(−x)`. |
| B2 | Interval ops, `divpos`, `rat`, `sqr` | PASS | P:38-56 and P:295-300: outward after every operation, NaN mapped to [−∞, ∞], and division only by an interval with strictly positive lower bound. The verifier has its own copy (V:31-49). |
| B3 | **Tiered Taylor order: \|mid\| ≤ 4 uses order 14 with REM_SMALL = 2e-14** | **PASS (doc N-P5b-1)** | Sine has degree 29, which is also its degree-30 Taylor polynomial (sine has only odd terms). Cosine has degree 28, which is also its degree-29 Taylor polynomial (cosine has only even terms). The Lagrange bound is therefore 4^30/30! = 4.35e-15 < 2e-14 for both. **The source comment is wrong:** it says "4^29/29! < 1.1e-14", but 4^29/29! = 3.26e-14 > REM_SMALL. The bound holds only through the degree+2 argument above; fix the comment. Exact-rational test: `test_small_order_remainder_is_valid`. For \|mid\| in (4, 8], order 30 with 1e-27 is unchanged (P5). The verifier uses order 30 everywhere (V:73). |
| B4 | Trig cache | PASS | P:125-137: `rnea(..., cqs, sqs)` uses the cached cos/sin of the SAME joint argument. The tm1 centre uses cos/sin(qc); dual use is cos/sin of the dual leaf box (P:414, 419); witnesses use cos/sin(pq) (P:448). |
| B5 | Interval span polynomials (built like K3) | PASS | P:369-393. Uniform cubic B-spline segment k is taken over t ∈ [t0, t1]; ×3, ×4 and ×6 of float32 values are exact in binary64. Timing uses the K3 table of exact (s − m/29)^p coefficients (`compiled_k3.timing_intervals`), so `compose(xt, L − m/29, h)` is the right variable. Derivatives are taken in global s (·1/h). Oracle agreement on Python's exact spans confirms the construction. |
| B6 | `to_bern`, `bsplit`, `elevate3`, `horner` | PASS (N-P5b-5) | P:307-326 implement the standard formulas with outward intervals: b_k = Σ C(k,i)/C(n,i)·p_i; de Casteljau left/right; elevation c_k = k/(m+1)·b_{k−1} + (1 − k/(m+1))·b_k. **N-P5b-5 (negligible):** `bsplit` halves with `s.lo*0.5`, `s.hi*0.5`, which is not outward only for subnormal inputs (magnitude ~1e-308). Any such slip is still caught by the verifier's recomputation. |
| B7 | tm1 leaf bound | PASS | P:410-432. The centre values at the exact dyadic centre; the dual slopes over Bernstein leaf hulls of q..q‴; the Bernstein hull of G0 + A0·x + ½B0·x′ (x is cubic and x′ is elevated to cubic); and the remainder (G1′ + A1′·X + ½B1′·X′)·[−H, H] with H = h·(b − a)/2 rounded outward. Same construction as the reviewed Python tm1 (P2 addendum). X ≥ 0 is required. |
| B8 | Witnesses at a / mid / b | PASS (N-P5b-3) | P:441-457. Point enclosures use interval Horner at exact dyadic points, and violation requires strict disjointness. VIOLATED_X is reported only when `px.hi < 0` (rigorous). **N-P5b-3 (low):** a torque witness is not prevented when the x enclosure merely straddles 0, i.e. x(u) ≈ 0 within rounding. The Python `verify_k4_witness` requires exact x ≥ 0, so recommend `px.lo >= 0` before a torque witness. Never observed. |
| B9 | Budgets, depth, invalid input, buffer retry | PASS | P:347-350 checks finite geometry, timing and limits, depth ≤ 30 and nodes ≥ 1. Each span is capped locally at `max_nodes` and the cumulative total is checked at merge (P:476-481). Leaf buffer overflow returns 7, which the wrapper retries with 4× capacity. Exceptions map to NUMERICAL_ERROR. |
| B10 | OpenMP merge and determinism | PASS | Spans are independent and merged in span order; the first failing span decides. The verifier works the same way (V:283-293). Producer and verifier results are identical across 1, 2, 3 and 8 threads on fixture boundary cases, an UNKNOWN_NODES budget case and 120 random runs (`test_thread_count_determinism`, campaign). |
| B11 | `bits_to_path`; witness u = (2·bits + where)/2^(depth+1) | PASS | Children are `bits<<1` (L) and `bits<<1 \| 1` (R), so the first decision is the MSB, as `bits_to_path` reads it. For 500 random (depth ≤ 30, bits) the decoded [a,b] equals bits/2^d and (bits+1)/2^d, the verifier's inverse (`replace` then base 2) round-trips, and the witness formula yields exactly a, mid and b for where = 0, 1, 2. |
| V1 | Verifier soundness | PASS | Own interval type; the same trig algorithm, separately coded at order 30 everywhere. Centred-form enclosures p(m) + p′([a,b])·(u − m) per leaf (V:156-160). Box torque G + A·X + ½B·X′ with world-frame dynamics: forward kinematics, R·Ic·Rᵀ and subtree sums (V:88-147). Requires X ≥ 0, finite values and limits; refines up to `refine` levels. Forged certificates were rejected 0 / 75 in the campaign, plus the boundary forgery at refine 8 and 20 (`test_verifier_rejects_forged_certificate_for_truly_violated_path`). |
| V2 | Verifier independence | PASS (shared items listed) | **Independent:** interval type and code; dynamics formulation (world frame vs link-frame RNEA); enclosure method (centred form plus box refinement vs Bernstein plus tm1); its own trig order. **Shared:** the path/timing construction algorithm (re-typed in each binary) and the K3 timing table; the contract-v2 model; and the Python `k4_interval` rpy enclosures inside the packed parameters (P2-reviewed). The shared path assumptions are covered by this review's oracle on `spline_polynomials` spans. |
| V3 | Exact dyadic partition | PASS | V:200-213. Every leaf requires 0 ≤ span < ns (ns derived by the verifier from C) and depth ≤ 30, bits < 2^depth. Leaves are mapped to integer segments on a 2^30 grid, sorted, and required to tile [0, 2^30] exactly for EVERY span. Rejected: missing, duplicate, parent-overlap, foreign or negative span, bad character, depth 31+, and a shifted leaf (`test_partition_and_digest_attacks`). |
| V4 | Recorded-box midpoint check | PASS (falsification) | V:242-253. The rigorous world-frame midpoint torque must lie in the recorded box (1e-9 relative slack), and the recorded box must lie inside the limits. Zeroed boxes and boxes beyond the limits are rejected. As in P2, recorded boxes are diagnostics; the verdict relies on the verifier's own enclosures. |
| V5 | Digest / contract binding | PASS (N-P5b-4) | KC:117-125 / 236-239: sha256 over the float32 geometry bytes, timing bytes, contract SHA and limit hex. It is checked before the C call, and the contract hash is checked against the frozen v2 SHA at construction. Rejected: an empty or wrong digest, one-ulp changes to geometry or timing, other limits, and a contract with gravity tampered. **N-P5b-4 (low, hardening):** the digest omits array shapes, and neither wrapper checks `timing.size == 32` or `geometry.shape[1] == dof`, while C reads `timing[0..31]`. A malformed call could read out of bounds. Recommend asserting the shapes and hashing them. |
| V6 | Physical-limit gate | NOTE (N-P5b-2) | `CompiledVerifier.verify` accepts any finite limits bound in the digest. It has no `expect_physical` counterpart to `verify_k4`, so the caller must pass the contract's limits when a physical claim is made (`test_verifier_has_no_physical_limit_gate`). |
| V7 | x < 0 / x = 0 | PASS | Producer: X.lo ≥ 0 is required for a leaf, and x(u) < 0 at a witness point gives VIOLATED_X (5 / 5 confirmed exactly, never CERTIFIED in random negative-timing cases). Verifier: X.l ≥ 0 is required. x = 0 is scope only, as before; K3 is required. |

## Findings

| ID | Severity | Finding | Recommended fix |
|---|---|---|---|
| N-P5b-1 | Doc | The REM_SMALL comment claims 4^29/29! < 1.1e-14, which is false (3.26e-14). The bound is valid through degree+2: 4^30/30! = 4.35e-15. | Correct the comment. Optionally assert with exact rationals at build time. |
| N-P5b-2 | Low | The compiled verifier has no physical-limit gate. | Add `expect_physical` equivalent to `verify_k4`. |
| N-P5b-3 | Low | A torque witness is allowed when the x enclosure straddles 0. | Require `px.lo >= 0` before a torque witness. |
| N-P5b-4 | Low | No shape checks; the digest omits shapes. | Check `timing.size == 32` and `geometry.shape[1] == dof` in both wrappers, and include the shapes in `bytes_digest`. |
| N-P5b-5 | Negligible | `bsplit` halving is not outward for subnormals. | Use `dn`/`up` after the halving. |

## Tests

- `tests/test_independent_k4_p5b.py`: 10 tests, OK, about 29 s. Libraries are built in a temp dir; the tests skip without g++ or mpmath.
- Campaign script: `run_campaign.py` (`PYTHONPATH=src:. python docs/reports/k4_p5b_independent_review/run_campaign.py`), about 190 s CPU.
- Full suite at a913a13 (working tree plus reviewer files): **395 tests OK (skipped=6)**, 264 s.

## SHA-256 (at a913a13)

```
721dca3713bc2c1057770310a053f7b1129f27cf4c743fb402e0c47ad135e26a  src/curobo_certify/native/k4_rnea.cpp
ab803d90e8a7dae2e38ceab6cb922411da32b3d0017a0d7c0882dc5f0bbe8943  src/curobo_certify/native/k4_verify.cpp
0542127766dfe153f98a3664fe487535de23009f6e7e0a7bd9414241cee97c29  src/curobo_certify/k4_compiled.py
21fef1d87576f2c10d95758e1e7f43fe17a7b22201361008a106fbb16824b1fa  tests/test_k4_compiled_pipeline.py
3701684abb24c298b92ae3570d4a4f401bb211b4c9382b98c49e354632392663  src/curobo_certify/compiled_k3.py (timing table; not re-reviewed)
9af0b96b6008dbb8ce58f7129b7ac5e1eed7c900649a7a5db728293f10aa95da  src/curobo_certify/spline_polynomials.py
15492e03b53c6681f5b52fa1e7ae647698688db95f661d1d83a0ffca766b655c  docs/reports/k4_p5b_independent_review/campaign.json
3c6ae8e97eef9156f444027951b552bde83363453cc62345fb6a2e3035d7c70d  docs/reports/k4_p5b_independent_review/run_campaign.py
a04c4afb2632bbd4972b0ad67c487b2c999a1730ad3bbb2ce8b03dc024252604  tests/test_independent_k4_p5b.py
```
