# Native time law: exhaustive float32 audit of time→progress inversion (Gate T1)

Date 2026-11-17, branch `exp/margin-duration-native-time-20261117`, upstream `NVlabs/curobo@c301cb00…`,
RTX 3090 (production GPU for all frozen outputs).

> **ERRATUM (2026-11-17, independent review R0, finding U1).** The "certified" ε_s below is NOT an
> upper bound: the kernel reduced the distance from s_native to the bracket (near end), a lower bound
> on |s_native − s*|. All ε values and L·ε state bounds in this document are withdrawn as bounds; the
> corrected audit (T1 v2, far-end distance) and the re-issued 224-path values are in
> `docs/T1_V2_CONTRACT_20261117.md`. Also withdrawn: the wording that native samples are samples of a
> trajectory "whose physical derivatives are the certified quantities" — reported v/a/j arrays are
> sampled quantities; no between-request interpolation contract exists. The text below is preserved
> unchanged as historical evidence.

## The question

The native sampler returns, for a requested time `t`, a progress `s_native(t)` (Newton iterations on a
511-cell FP32 time table + 6-point Gauss quadrature, clamped) and reports `q, v, a, j` evaluated at that
progress. Earlier work bounded the reported values at a *given* progress (NATIVE_REPORTED_K3, conditional
on FP32 semantics A1–A5) but left open whether `s_native(t)` is related to the exact inverse `s*(t)` of the
mathematical time law `T(s) = ∫_0^s dσ/√x(σ)` — i.e. whether the native samples are samples of a
well-defined continuous-time trajectory whose physical derivatives are the certified quantities.

Bitwise equality of a float32 `s_native(t)` with a smooth `s*(t)` is impossible and not required. The
useful contract is a certified bound `ε_s ≥ |s_native(t) − s*(t)|` for every admissible requested time.

## Key observation and method

The native kernel only ever receives **float32** requested times. `t ↦ s_native(t)` is therefore a function
on a finite set (≈1.05·10⁹ values in `[0, T_native]` for a 0.3 s path). It can be enumerated exhaustively on
the GPU, executing the pinned upstream `progress_at_time` compiled with cuRobo's own NVRTC options.

1. **Exact time law, rigorously (CPU, `src/curobo_certify/time_law_grid.py`).** On `2^22` dyadic cells:
   midpoint rule with remainder `h³/24·sup|f''|`, `f = x^{-1/2}`, `f'' = ¾x^{-5/2}x'² − ½x^{-3/2}x''`, all
   with outward-rounded interval Horner on each cell; cells containing a timing knot use a hull bound;
   recursive-summation error `γ_N Σ|t_i|` added explicitly. Output: `T_lo[k] ≤ T(k/N) ≤ T_hi[k]` and
   per-cell `fmin ≤ 1/√x ≤ fmax`. Requires certified `x > 0` (true on all 224 real outputs: TimeOpt
   enforces rest through stationary geometry, x(0), x(1) ≥ 1.5).
2. **Rigorous inverse bracket.** For `t`, with `k = max{k: T_hi[k] ≤ t}` and `j = max{j: T_lo[j] < t}`:
   `s* ≥ min(s_k + (t − T_hi[k])/fmax_k, s_{k+1})`, `s* ≤ min(s_j + (t − T_lo[j])/fmin_j, s_{j+1})`
   (monotonicity and per-cell slope bounds). After the model duration the continuous trajectory holds its
   end state (`s* = 1`).
3. **Exhaustive GPU audit (`src/curobo_certify/native/time_inverse_audit.cuh`,
   `scripts/native_time_inversion_audit.py`).** For every float32 bit pattern from 0 to `T_native`:
   production `s_native(t)`, bracket lookup with CUDA directed-rounding double intrinsics, atomic max of the
   certified distance. A CPU recomputation on 5,000 random times cross-checks the GPU reduction.
4. **Exhaustive reported-value audit (`reported_audit` kernel).** For every same float32 time, the
   production outputs (progress + sample tail, the expressions of `sample_with_progress_kernel`) are
   checked against the physical limits: `max |v|/v_lim`, `|a|/a_lim`, `|j|/j_lim`, non-finite count.

Strict-rest endpoints with `x = 0` (not present in real outputs) are handled in the exact Fraction
module by root substitution `u = w²` (simple root → regular integrand `2/√(g(w²))`; order ≥ 2 →
NONINTEGRABLE_ENDPOINT, negative slope → NEGATIVE_X_AT_ENDPOINT); tests compare with closed forms.

## Results (224 real outputs: 145 margin sweep + 79 earlier; 2.364·10¹¹ float32 times; RTX 3090)

- All 224 time-law grids certified (`x > 0`); model durations bracketed to ≈3·10⁻¹⁰ s; native durations
  differ from the model by ≈10⁻⁸ s.
- **Certified `max_t |s_native(t) − s*(t)|` ≤ 2.44·10⁻⁷** (median over paths 1.46·10⁻⁷), i.e. a few
  float32 ulps of `s`; all CPU cross-checks consistent.
- Induced shift of the exact trajectory's state at the same physical time (`L·ε_s`, rigorous Bernstein-hull
  Lipschitz bounds): velocity ≤ 8.0·10⁻⁶, acceleration ≤ 8.5·10⁻⁶, jerk ≤ 2.6·10⁻⁴ of the physical limits
  (jerk away from geometry knots; within ε_s of a knot the native jerk is the exact trajectory's one-sided
  jerk on the other side of the knot — both one-sided values are certified by the K3 proof).
- **Reported values at every float32 time: 200/224 within physical limits, 0 non-finite.** All 24
  exceedances are scale-1.00 outputs (12 margin sweep + 12 earlier margin arm), max jerk 1.00355× limit;
  they coincide exactly (24/24, no model-only or native-only case) with the outputs whose exact serialized
  model has a genuine violation witness.

## Execution contract established (scope NATIVE_PHYSICAL_TIME_LINKED, per path)

For every float32 requested time `t ∈ [0, T_native]`, the native sample `(q, v, a, j)(t)` is within
`E_same-progress + L·ε_s` of the state of the continuous-time trajectory `q*(t) = q(s*(t))` of the exact
serialized model, whose K3 constraints are certified (or violated with an exact witness) for all real `t`.
Assumptions: (i) the audit harness executes the same compiled upstream code as production — supported by
bitwise reproduction of saved production samples on all 224 paths; (ii) GPU determinism; (iii) for
`E_same-progress` only, A1–A5 (now checked against the SASS instruction stream and, for reciprocal,
constant division, sqrtf and rsqrtf, exhaustively measured on all positive normal float32 inputs). The
exhaustive reported-value result needs neither (iii) nor the model.

Not established: behaviour between float32 times beyond what the controller does with samples (no
interpolation contract exists in the native API); robot tracking; torque/torque-rate; other GPUs or
compiler versions (re-audit required); formal verification of the audit code.

Gate T1: **PASS for the scoped contract above, exploratory** (conditions stated). The previous blocker
`BLOCKED_NATIVE_TIME_INVERSION` is replaced by this per-output audited contract; it is not a universal
proof for unseen outputs — each output must be audited (≈5 s on an RTX 3090 per path, plus ≈10 s CPU),
which is far from online latency.
