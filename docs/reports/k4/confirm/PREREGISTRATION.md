# K4-Panda-Confirm150 — preregistration (frozen before any confirmation solve)

Date: 2026-11-18. Branch `exp/k4-certified-torque-reference-20261117`. Written after the exploratory studies
(K4-Panda-Dev40, PayloadDev40, CorrectedDev40, RegimeDev40, TOPP-RA baselines), which informed the hypotheses
below. None of the confirmation queries had been solved when this file was committed.

## Population
- `scripts/k4_confirm_population.py freeze`.
  - Fresh seed (SHA-256 of `curobo_certify/k4_panda_confirm_v1/K4-Panda-Confirm150`).
  - Same generator distribution as Dev40: 4 scenes × 3 motion magnitudes, HOME start.
  - Pool: 230 ordered queries, leakage-audited against every prior Panda query file.
- Conditions, all at the 0.95 TimeOpt margin. **First-K = 150** successes per condition over the frozen pool order. Failures are retained and nothing is replaced.

| condition | planning model | payload | TimeOpt grid |
|---|---|---|---|
| CA_CUROBO | cuRobo | default (0.01 kg) | 511 |
| CB_CUROBO_3KG | cuRobo | hypothetical 3 kg (hook) | 511 |
| CC_CORR_3KG | identified + URDF hand (hook) | hypothetical 3 kg | 511 |
| CD_CORR_3KG_NI063 | identified + URDF hand | hypothetical 3 kg | 63 |

- TOPP-RA baseline: toppra 0.6.3, trimmed moving segment, default gridpoints, the same velocity, acceleration and 0.95 torque limits. It runs on the CC geometries (`k4_toppra_run.py --trim --grids auto`).
- If a condition's pool is exhausted before K, it is reported with its true denominator.

## Measurements (per eligible output)
- **MODEL_K3:** compiled producer + verifier.
- **MODEL_K4 and PLANNING_K4** under the condition's planning model:
  - tm1 + independent verifier at protocol refinement 8;
  - a verifier non-confirmation is REJECTED_BY_VERIFIER, never certified;
  - an escalation to refinement 12 is reported separately.
- **IDENTIFIED_K4** (identified + hand, + payload where present) for every condition.
- **Rigorous max-utilisation bracket** [lb, ub] at the planning limits (`k4_baselines.py`), plus exact-model sampled checks and native float32 sampled grids (TimeOpt grid, S8193, S32768).
- **Violation magnitude:** excess = lb − 1 (rigorous lower bound). Reported at the thresholds 1e-6, 1e-4, 1e-3 and 1e-2. **A "meaningful" violation is excess ≥ 1e-4** (0.01% of the limit). Smaller excesses are reported as tolerance-level.

## Hypotheses and decision rules
| id | statement | test / rule |
|---|---|---|
| H1 soundness | No output is verified CERTIFIED for a contract while that contract is rigorously violated. Concretely: no lb > 1 with a verified certificate, no native dense sample above the limit by more than 1e-4 relative, and no disagreement between the two independent verifiers after escalation. | Any instance → **HARD_STOP**: stop, preserve evidence, report. |
| H2 coverage | MODEL_K4 under the planning model at physical limits, in CA and CC. | Exact one-sided 95% lower bound ≥ 0.95 → GO. |
| H3 model mismatch | In CB, the proportion violating IDENTIFIED_K4 (physical limits). | Exact two-sided 95% CI reported. "Mismatch is a real risk" is claimed iff the lower bound > 0.05. |
| H4 fix | In CC, the proportion violating IDENTIFIED_K4. | Exact upper 95% bound reported. Claim "corrected planning removes it" iff 0 violations; the H3 vs H4 difference is tested with an exact one-sided test on the paired subset, p < 0.05. |
| H5 coarse grid | Meaningful PLANNING_K4 between-sample violations: CD vs CC, paired by query. | Exact one-sided McNemar test, p < 0.05. The CD rate is reported with its CI. |
| H6 dense sampling | Agreement of the native S32768 verdict with the rigorous verdict (VIOLATED vs CERTIFIED; UNDECIDED excluded and counted), pooled over CA–CD. | Disagreements reported with an exact CI. Any meaningful miss (excess ≥ 1e-4 not seen by S32768) is reported individually. |
| H7 TOPP-RA | The proportion of TOPP-RA default-grid parametrizations with a meaningful violation of their own torque limit. | Exact CI. Claimed "systematic" iff the lower bound ≥ 0.25. |
| H8 enforcement grid | Meaningful PLANNING_K4 violations missed by TimeOpt's own enforcement grid, in CC (511) and CD (63). | Rates and exact CIs (descriptive). |

Secondary (descriptive only):
- durations for CC vs CB (paired);
- native failure rates per condition;
- K4 cost (producer + verifier) vs GPU sampled-check cost;
- per-scene and per-magnitude breakdowns.

## Fixed analysis code
- `scripts/k4_certify_outputs.py`, `k4_baselines.py`, `k4_native_torque_samples.py`, `k4_toppra_run.py` / `k4_toppra_certify.py`, `k4_analyze_baselines.py`, `k4_analyze_extensions.py`, as committed with this file.
- Any later change is reported as a deviation with its reason.
