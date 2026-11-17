# K4 P0: Panda dynamics model freeze and independent parity

Date 2026-11-17. Branch `exp/k4-certified-torque-reference-20261117`.

## What cuRobo actually uses

`scripts/k4_extract_curobo_model.py` builds the planner exactly as the native generator does (`franka.yml`, pinned `c301cb00`). It runs one warm-up plan and reads the TimeOpt optimizer's own `KinematicsParams` and `Dynamics` objects.

**Rigid-body tree**
- 13 links: `panda_link0` … `panda_link7`, `panda_link8`, `panda_hand`, the two fingers, and `attached_object`.
- Fixed base, identity world frame.
- 7 Z-axis revolute joints.
- Finger joints locked at 0.04 m and baked into fixed transforms.
- cuRobo lumps fixed links into their movable ancestors numerically, in float64, then stores float32.

**Parameters**
- float32 values of the pinned URDF inertials for `panda_link1`…`panda_link7` and `panda_hand`.
- cuRobo default inertials (0.01 kg, I = 1e-4) for links with no URDF inertial: `panda_link8`, both fingers, and `attached_object`.

**Gravity:** `DynamicsCfg` default (0, 0, −9.81), stored as float32.

**TimeOpt torque handling**
- Effort limits: ±87 N·m on joints 1–4 and ±12 N·m on joints 5–7.
- TimeOpt plans against `limit_scale` × effort: ±82.65 and ±11.4 N·m at 0.95.
- It checks torque only at its constraint grid: 512 output samples, both sides of every path knot, and the timing breakpoints.
- Each torque uses the coefficient identity with three inverse-dynamics calls per grid point.

**Frozen contract:** `docs/reports/k4/p0/panda_model_contract.json`. The `contract_sha256` = `670b1e7ed80489dc06c21245f613fcb16148475b6390e8a9e3462279ee0c179f` covers the canonical model JSON, with every parameter stored as exact float32 hex. The contract also records:
- `franka.yml` SHA-256 `91d48371…`;
- URDF SHA-256 `8f4d6768…`;
- extraction NPZ SHA-256 `0b154add…`.

Scope label: **`CUROBO_PINNED_PANDA_RIGID_BODY_MODEL`**.

## Provenance of physical constants

| Item | Source | Status |
|---|---|---|
| Joint frames and axes | pinned URDF / `franka.yml` | Frozen. Pinocchio's independent URDF parse agrees (below). |
| Link masses, CoMs and inertias (links 1–7, hand) | pinned URDF `<inertial>` blocks | Origin not documented in the repository. Physically inconsistent (below). |
| Inertials of link8, fingers and `attached_object` | cuRobo default (0.01 kg, 1e-4) | Injected by cuRobo, not physical. |
| Torque limits | URDF `effort`, matching the manufacturer's FER table (τ_max 87, 87, 87, 87, 12, 12, 12 N·m, https://frankarobotics.github.io/docs/robot_specifications.html) | **Sourced.** |
| Torque-rate limit (1000 N·m/s) | Manufacturer table | Not modelled by cuRobo. **Out of scope.** |
| Friction, damping, rotor inertia, motor limits | None in cuRobo | **Unsupported.** `ROBUST_ACTUATOR_TORQUE` is not claimed. |
| Payload of 3 kg or 5 kg | None: no sourced mass, CoM, inertia, collision geometry or attachment | **`BLOCKED_PAYLOAD`.** |

## Physical-validity audit

These findings are reported, never repaired:

- **`panda_link3`** (affects joint torques): the CoM inertia is **not positive definite**, with eigenvalues −0.003811, 0.003253 and 0.021659. No rigid body has this inertia.
- **`panda_link4`:** its inertia tensor is identical to `panda_link1`'s.
- **`panda_hand`:** its inertia tensor is identical to `panda_link7`'s, with a round 1.0 kg mass.
- **Default inertials:** link8, both fingers and `attached_object` carry cuRobo's injected 0.01 kg defaults.
- **`panda_link0`:** the inertia is not positive definite, but the link is fixed, so it has no effect on joint torques.

Status: **`PHYSICALLY_INCONSISTENT_PARAMETERS`.**

## Independent numerical parity (diagnostic, not a proof)

`scripts/k4_p0_parity.py` gives the result in `docs/reports/k4/p0/parity.json`. It compares three implementations on 4 × 128 seeded states: gravity-only, full states, B probes and A probes.

1. **Algorithm check**
   - Pinocchio 4.1.0 parses the URDF independently, with locked fingers and cuRobo's inertial set configured.
   - Its float64 RNEA lies **inside** this repository's rigorous interval RNEA enclosure on all 512 of 512 states.
   - The enclosure was computed on Pinocchio's own parameterization, and the largest difference is 8.5e-14 N·m.
   - This validates frames, axes, signs, gravity and the recursion.
2. **Parameter precision:** the URDF float64 decimals differ from cuRobo's stored float32 parameters by at most 5.8e-7 relative. The difference comes from float32 rounding of the inertials, transforms and gravity (9.81f), as the float32 attribution test shows.
3. **Native cuRobo GPU:** float32 torques lie within 3.7e-6 relative (at most 1.9e-5 N·m) of the exact-model enclosure. This is a diagnostic only; the native error analysis is P5.
4. **Coefficient identity:** cuRobo's G = F(q,0,0), B = (F(q,0,q′) − G)/2 and A = F(q,q′,q″) − G agree with the directly computed interval A, B and G to 1.4e-5. They are exact in the model, since it has no friction or damping terms. The interval check confirms that τ = A + G overlaps at x = 1, x′ = 0.

Verdict: **PARITY_PASS.**

## P0 decision

- **Model contract:** FROZEN as `CUROBO_PINNED_PANDA_RIGID_BODY_MODEL`. This is the model cuRobo TimeOpt enforces, so certifying against it answers whether cuRobo's own torque constraints hold between its samples.
- **Torque limits:** SOURCED.
- **Physical Panda torque fidelity:** **`BLOCKED_MODEL_PHYSICAL_VALIDITY`.** The inertials are physically inconsistent, duplicated or injected, and their provenance is undocumented. No K4 result may be called a physical Panda torque certificate. Every certificate carries the cuRobo-model scope.
- **Payloads of 3 kg and 5 kg:** **`BLOCKED_PAYLOAD`.** No payload experiments are run. The baseline condition is cuRobo's default `attached_object` (0.01 kg), labelled `PAYLOAD_DEFAULT_0P01KG`.

## Amendment: contract v2

**Finding from building the independent verifier, P2.** The link-frame RNEA and the world-frame verifier both read v1, cuRobo's float32 numbers taken exactly. Their rigorous enclosures did not intersect: the midpoints differed by 3–8e-8 N·m, while the widths were about 1e-12. The cause is that cuRobo's stored float32 fixed rotations are not orthonormal:

- `panda_hand` is off by 3.4e-8;
- the composed locked-finger transforms are off by 9.5e-7.

v1 therefore did not define a rigid-body model. Two correct formulations of it legitimately disagree.

**Fix: contract v2.** `docs/reports/k4/p0/panda_model_contract_v2.json`, SHA `9caa38bff95b56df89ea6dc71a08a5558474e764a613ecc480c4c38364f663e7`. It supersedes v1 (`670b1e7e…`), which stays in git history.

- Every value is the exact decimal from the pinned URDF, or cuRobo's documented default (0.01 kg, 1e-4; finger lock 0.04 m; gravity 9.81), read as an exact rational.
- Rotations are the exact URDF rpy rotations R = Rz(yaw)·Ry(pitch)·Rx(roll), enclosed with interval trigonometry.
- `scripts/k4_model_contract_v2.py` cross-checks every value against cuRobo's stored float32:
  - translations, masses, CoMs and inertias are equal in float32 for every link;
  - the finger translations differ by 2.5e-8, from cuRobo's float32 composition;
  - rotation differences are at most 1.2e-8 for the hand and 4.8e-7 for the fingers, and 8e-20 elsewhere.
- cuRobo's float32 representation is thereby part of the native numerical error.

**Result:** with v2, the producer (link frame) and the verifier (world frame, explicit Jacobian-transpose sums) intersect on 200 random states, with and without gravity. The largest midpoint gap is 2e-13; the enclosures are about 1.5e-11 wide. `tests/test_k4_path.py` checks this again on 40 states.
