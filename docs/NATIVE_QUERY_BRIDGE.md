# Native source-query bridge: real-path pilot

## Pinned API and conversion contract

The bridge uses cuRobo `MotionPlanner` from NVlabs/curobo commit
`c301cb00a87012c2808b1a60649c4065c9b3c95a` with `enable_trajopt=True`,
`enable_timeopt=True`, and native inverse dynamics enabled. `MotionPlanner`
first builds and validates a collision-free geometric path, then its native
`TimeOptSolver.timeopt` path creates the returned `SplineTrajectory`. The
bridge clones `result.get_interpolated_plan().spline_trajectory` after
selection. It exports the effective complete cubic cardinal control polygon
and the exact 32 `TimeOptTiming.controls`; it does not fit coefficients from
sampled trajectories or reinterpret cached quintic controls. Subdivision is
only a certifier representation change.

The serialized query source contains the robot, start/goal joint states,
scene config, deterministic seed, query ID and SHA256. The output retains robot
and scene config hashes, effective geometry and timing hashes, solver status,
duration, limits, cloned state samples, and the upstream SHA. Source scenes
are the pin's empty, collision-table, and collision-test examples. Goal
generation uses deterministic small joint-space deltas; planner rejection is
retained, not replaced.

The selected result's cubic spatial curve is the geometric path TimeOpt
actually retimes. The bridge records `conversion: none` relative to that
selected TimeOpt spline. It does not claim that this path equals a separately
cached pre-TimeOpt or quintic TrajOpt representation.

## Environment compatibility

The exact pinned source runs in conda environment `curobo-timeopt-c301cb`
(Python 3.11.17, PyTorch 2.5.1+cu124, CUDA runtime 12.4, RTX 3090). The
upstream code expects `torch.cuda.get_device_properties().shared_memory_per_block`,
which the installed PyTorch `DeviceProperties` does not expose. A runtime-only
proxy obtains shared-memory limits from `libcudart.so.12`; measured values are
49,152 bytes and 101,376 opt-in bytes. Neither upstream source nor TimeOpt
algorithm is modified. The exact environment capture is
`runs/native_environment.json` when present.

## Pilot results and blockers

The frozen 16-row source-request manifest is `runs/native_pilot_queries.json`
(seed 20261110; SHA256
`a74a635476d9442c0137136e4e414cb610c79542537af99baab573aa34025767`). It
contains explicit absolute Panda start/goal positions before planning. The
earlier delta-only version is preserved as
`runs/native_pilot_queries_initial_unresolved.json`. Eight Panda planner calls all
returned native-success trajectories from empty/table/test scenes. The eight
UR5e rows are retained as `FAILED_RETAINED`: the pinned repository has no
verified UR5/UR5e robot YAML/URDF pair. Its available `ur10e.yml` is not treated
as UR5. A caller can provide a verified external config, but there is no such
asset in this run.

All eight Panda adapter checks exercised 1,001 random interior times plus both
sides of 46 common path/timing knots and compared positions, path derivatives,
physical velocity, acceleration and jerk. Position, velocity, acceleration,
and path derivatives pass the declared 4e-4 tolerance; physical jerk fails on
each path. The worst absolute jerk discrepancy is 0.005144 on Panda query 06
(native -96.82724, polynomial adapter -96.82210, joint 0, t=0.113864 s).
No tolerance was widened. These are native successes but adapter-invalid
outputs; no native-linked certificate may be counted.

For diagnosis only, the exact serialized polynomial model was checked with
strict-rest selected. The native endpoint `x` is not zero, but physical endpoint
velocities are zero because the clamped spatial path has zero endpoint tangent;
strict rest also requires exact zero joint acceleration. The model satisfies
exact endpoint q, exact knot continuity/regularity, local duration and K3 proof
obligations at max depth 0, with 329 proof nodes/leaves per trajectory and
independent proof replay passing. This is a **conditional
model-level K3 result**, not a certificate for native physical execution,
because adapter jerk parity fails. Depths 2, 4, 8, 12 and 16 produce the same
proof, with no subdivision. Torque, torque-rate, collision, and full dynamics
remain unsupported. Strict-rest conditions were not selected and are not
claimed.

The native S8193 and event-aware S32768 arrays are preserved in each NPZ. A
separate corrected event-audit run checked actual timing progress on both sides
of all 46 knots (92/92 sides verified) for Panda query 00; the adapter parity
check also evaluates all 92 one-sided knot times on every successful path.
These grids are diagnostics only. Maximum planning-scaled diagnostic utilization is
below one in these outputs; this does not repair the parity failure and does
not constitute proof.

## Reproduction

```bash
python scripts/freeze_native_queries.py --out runs/native_pilot_queries.json
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src conda run -n curobo-timeopt-c301cb \
  python scripts/generate_native_pilot.py --queries runs/native_pilot_queries.json \
  --out-dir runs/native_pilot_resolved --robot panda --limit 8
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src conda run -n curobo-timeopt-c301cb \
  python scripts/generate_native_pilot.py --queries runs/native_pilot_queries.json \
  --out-dir runs/native_pilot_resolved --robot ur5e --limit 8
PYTHONPATH=src python scripts/certify_real_pilot_models.py \
  --pilot-dir runs/native_pilot_resolved \
  --out runs/native_pilot_resolved/k3_model_diagnostics.json
python scripts/build_real_pilot_report_data.py \
  --pilot-dir runs/native_pilot_resolved --out runs/native_pilot_resolved/report_pilot.json \
  --event-audit runs/native_pilot_event_audit/summary.json
python tools/build_report.py --pilot runs/native_pilot_resolved/report_pilot.json
```

Pilot artifacts live under ignored `runs/` and are intentionally not committed.
No Dev40/Holdout100/Confirm300 corpus, paired latency experiment, bootstrap
uncertainty, or fresh-request measurement is present. Do not use this pilot to
tune against holdout or to claim campaign acceptance.
