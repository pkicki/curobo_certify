#!/usr/bin/env python3
"""Run deterministic native-vs-adapter parity on a one-joint smoke path.

The adapter evaluates the serialized cubic control polygons independently. It
does not copy cuRobo kernels; the pinned upstream remains the native oracle.
This smoke path is engineering validation, not a campaign robot/query sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad

from curobo_certify.adapters.timeopt import (
    UPSTREAM_SHA,
    clone_effective_trajectory,
    require_pinned_upstream,
    sha256_array,
)


def _digest(value: torch.Tensor | np.ndarray) -> str:
    if isinstance(value, torch.Tensor):
        return sha256_array(value)
    array = np.ascontiguousarray(value)
    return hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()).hexdigest()


def _path_polynomial(points: np.ndarray, spans: int, s: float) -> tuple[np.ndarray, ...]:
    """Evaluate exact-rational cardinal cubic basis coefficients in float64."""
    z = min(max(float(s), 0.0), 1.0) * spans
    segment = min(int(z), spans - 1)
    u = z - segment
    support = points[segment : segment + 4].astype(np.float64)
    # Rows are descending powers u^3 ... u^0; entries are exact sixths.
    matrix6 = np.asarray(
        [[-1, 3, -3, 1], [3, -6, 0, 4], [-3, 3, 3, 1], [1, 0, 0, 0]],
        dtype=np.float64,
    ) / 6.0
    # Each row is one basis polynomial; sum support_i * basis_i.
    descending = (support.T @ matrix6)[0]
    c0, c1, c2, c3 = descending[::-1]
    q = ((c3 * u + c2) * u + c1) * u + c0
    qs = ((3 * c3 * u + 2 * c2) * u + c1) * spans
    qss = (6 * c3 * u + 2 * c2) * spans**2
    qsss = 6 * c3 * spans**3
    return q, qs, qss, qsss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/native_adapter_parity.json"))
    parser.add_argument("--arrays-out", type=Path, default=Path("runs/native_adapter_parity_arrays.npz"))
    parser.add_argument("--gpu-name", default="NVIDIA GeForce RTX 3090")
    args = parser.parse_args()
    require_pinned_upstream()
    from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
    from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
    from curobo.kinematics import KinematicsCfg
    from curobo._src.types.device_cfg import DeviceCfg

    torch.manual_seed(20261110)
    np.random.seed(20261110)
    if not torch.cuda.is_available():
        raise RuntimeError("native TimeOpt requires CUDA")
    device_index = next((i for i in range(torch.cuda.device_count())
                         if torch.cuda.get_device_name(i) == args.gpu_name), None)
    if device_index is None:
        raise RuntimeError(f"requested GPU {args.gpu_name!r} unavailable")
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    urdf = """<robot name="parity_pendulum"><link name="base"/>
<link name="body"><inertial><origin xyz="0 0 0.5"/><mass value="1"/>
<inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
</inertial></link><joint name="hinge" type="revolute"><parent link="base"/>
<child link="body"/><axis xyz="0 1 0"/>
<limit lower="-3" upper="3" velocity="2" effort="2"/></joint></robot>"""
    with tempfile.TemporaryDirectory(prefix="timeopt-parity-") as tmp:
        urdf_path = Path(tmp) / "parity.urdf"
        urdf_path.write_text(urdf)
        robot = KinematicsCfg.from_basic_urdf(
            str(urdf_path), "base", ["body"], device_cfg=DeviceCfg.from_basic("cuda", device_index)
        )
        solver = TrajectoryTimeOpt(
            TrajectoryTimeOptCfg(
                robot,
                batch_size=1,
                n_knots=16,
                timing_controls=32,
                num_intervals=511,
                max_iterations=10,
                acceleration_limits=torch.tensor([3.0], device=device),
                jerk_limits=torch.tensor([50.0], device=device),
                use_cuda_graph=True,
                enable_inverse_dynamics=False,
            )
        )
        start = torch.tensor([[0.0]], device=device)
        goal = torch.tensor([[0.2]], device=device)
        controls = torch.linspace(0, 0.2, 16, device=device)[None, :, None].contiguous()
        solver.timeopt(controls, start, goal)
        torch.cuda.synchronize(device)
        retained, trajectory = clone_effective_trajectory(solver)
        if not bool(retained.success[0]):
            raise RuntimeError(f"native TimeOpt failed status={retained.status.tolist()}")

        points = trajectory.control_points[0].detach().cpu().numpy().astype(np.float64)
        timing = trajectory.timing.controls[0].detach().cpu().numpy().astype(np.float64)
        spans = points.shape[0] - 3
        timing_spline = solver._basis.spline
        duration = float(retained.duration[0].item())
        if not duration > 0 or not np.isfinite(duration):
            raise RuntimeError(f"unexpected duration {duration}")
        x_spline = __import__("scipy.interpolate", fromlist=["BSpline"]).BSpline(
            timing_spline.t,
            timing,
            timing_spline.k,
            extrapolate=False,
        )

        def time_at_progress(s: float) -> float:
            if s <= 0:
                return 0.0
            return quad(lambda z: 1.0 / math.sqrt(float(x_spline(z))), 0.0, s, epsabs=1e-11, epsrel=1e-11, limit=200)[0]

        random_times = np.random.uniform(1e-7 * duration, (1 - 1e-7) * duration, 1001)
        knots = sorted(set([k / spans for k in range(1, spans)] + list(solver._basis.breaks_cpu[1:-1])))
        sides: list[float] = []
        for knot in knots:
            epsilon = min(1e-6, knot / 4, (1 - knot) / 4)
            for s in (knot - epsilon, knot + epsilon):
                t = time_at_progress(s)
                if 0 < t < duration:
                    sides.append(t)
        sample_times = torch.as_tensor(np.r_[random_times, sides], dtype=torch.float32, device=device)[None].contiguous()
        native, progress = solver._sample(sample_times)
        torch.cuda.synchronize(device)
        native = native.detach().cpu().numpy().astype(np.float64)[:, 0, :, 0]
        progress = progress.detach().cpu().numpy().astype(np.float64)[0]

        x_ref = np.asarray(x_spline(progress), dtype=np.float64)
        x1_ref = np.asarray(x_spline(progress, 1), dtype=np.float64)
        x2_ref = np.asarray(x_spline(progress, 2), dtype=np.float64)
        geometry = np.asarray([_path_polynomial(points, spans, s) for s in progress])
        q_ref, qs_ref, qss_ref, qsss_ref = [geometry[:, k] for k in range(4)]
        speed = np.sqrt(np.maximum(x_ref, 0))
        v_ref = qs_ref * speed
        a_ref = qss_ref * x_ref + 0.5 * qs_ref * x1_ref
        j_ref = speed * (qsss_ref * x_ref + 1.5 * qss_ref * x1_ref + 0.5 * qs_ref * x2_ref)
        eps = np.finfo(np.float64).tiny
        recovered_qs = native[1] / np.maximum(speed, eps)
        recovered_qss = (native[2] - 0.5 * recovered_qs * x1_ref) / np.maximum(x_ref, eps)
        recovered_qsss = (native[3] / np.maximum(speed, eps) - 1.5 * recovered_qss * x1_ref - 0.5 * recovered_qs * x2_ref) / np.maximum(x_ref, eps)

        pairs = {
            "position": (native[0], q_ref),
            "path_q_s_recovered_from_native_velocity": (recovered_qs, qs_ref),
            "path_q_ss_recovered_from_native_acceleration": (recovered_qss, qss_ref),
            "velocity": (native[1], v_ref),
            "acceleration": (native[2], a_ref),
            "jerk": (native[3], j_ref),
            "path_q_sss_recovered_from_native_jerk": (recovered_qsss, qsss_ref),
        }
        errors = {}
        for name, (actual, expected) in pairs.items():
            delta = np.abs(actual - expected)
            rel = delta / np.maximum(1.0, np.abs(expected))
            errors[name] = {"max_abs": float(delta.max()), "max_scaled_rel": float(rel.max())}
        # Explicit consistency threshold for the FP32 native path/timing kernels.
        tolerance = {"max_abs": 4e-4, "max_scaled_rel": 4e-4}
        failed = [name for name, e in errors.items() if e["max_abs"] > tolerance["max_abs"] or e["max_scaled_rel"] > tolerance["max_scaled_rel"]]

        args.arrays_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.arrays_out,
            requested_controls=controls.detach().cpu().numpy(),
            requested_start=start.detach().cpu().numpy(),
            requested_goal=goal.detach().cpu().numpy(),
            effective_geometry=trajectory.control_points.detach().cpu().numpy(),
            timing_controls=trajectory.timing.controls.detach().cpu().numpy(),
            native_time_grid=retained.time.detach().cpu().numpy(),
            native_position=retained.joint_state.position.detach().cpu().numpy(),
            native_velocity=retained.joint_state.velocity.detach().cpu().numpy(),
            native_acceleration=retained.joint_state.acceleration.detach().cpu().numpy(),
            native_jerk=retained.joint_state.jerk.detach().cpu().numpy(),
            physical_limits=np.asarray([[2.0], [3.0], [50.0]], dtype=np.float32),
            planning_scaled_limits=np.asarray([[1.9], [2.85], [47.5]], dtype=np.float32),
        )
        arrays_sha256 = hashlib.sha256(args.arrays_out.read_bytes()).hexdigest()

        def dense_check(sample_count: int, event_aware: bool) -> dict:
            event_targets = []
            if event_aware:
                dense_s = np.linspace(0.0, 1.0, sample_count)
                for knot in knots:
                    epsilon = min(1e-4, knot / 4, (1 - knot) / 4)
                    for s in (knot - epsilon, knot + epsilon):
                        index = int(round(s * (sample_count - 1)))
                        dense_s[index] = s
                        event_targets.append((float(knot), float(s)))
                dense_s.sort()
                output_grid = np.linspace(0.0, 1.0, retained.time.shape[1])
                sample_seconds = np.interp(dense_s, output_grid, retained.time[0].detach().cpu().numpy())
            else:
                dense_s = None
                sample_seconds = np.linspace(0.0, duration, sample_count)
            values, observed_progress = solver._sample(
                torch.as_tensor(sample_seconds, dtype=torch.float32, device=device)[None].contiguous()
            )
            torch.cuda.synchronize(device)
            values = values.detach().cpu().numpy().astype(np.float64)[:, 0, :, 0]
            observed_progress = observed_progress.detach().cpu().numpy().astype(np.float64)[0]
            physical_limits = (2.0, 3.0, 50.0)
            planning_limits = tuple(v * 0.95 for v in physical_limits)
            ratios = {}
            for label, limits in (("physical", physical_limits), ("planning_scaled", planning_limits)):
                family = {}
                for index, name, limit in zip((1, 2, 3), ("velocity", "acceleration", "jerk"), limits):
                    ratio = np.abs(values[index]) / limit
                    family[name] = {"max_sampled_ratio": float(np.nanmax(ratio)),
                                    "sampled_points_over_limit": int(np.count_nonzero(ratio > 1.0))}
                ratios[label] = family
            event_coverage = []
            if event_aware:
                for knot in knots:
                    below = observed_progress[observed_progress < knot]
                    above = observed_progress[observed_progress > knot]
                    event_coverage.append({
                        "knot": float(knot),
                        "nearest_left_delta": float(knot - below.max()) if below.size else None,
                        "nearest_right_delta": float(above.min() - knot) if above.size else None,
                    })
            observed_sides = sum(1 for row in event_coverage
                                 if row["nearest_left_delta"] is not None and row["nearest_right_delta"] is not None)
            return {"evidence_kind": "SAMPLED_DIAGNOSTIC_ONLY", "count": sample_count,
                    "grid": "uniform_physical_time" if not event_aware else
                            "uniform_progress_with_both_sides_of_each_union_knot_injected",
                    "requested_knot_side_targets": event_targets,
                    "observed_knots_with_samples_on_both_sides": observed_sides,
                    "total_interior_knots": len(knots) if event_aware else 0,
                    "observed_knot_side_coverage_pass": (observed_sides == len(knots)) if event_aware else None,
                    "constraint_samples": ratios,
                    "is_certificate": False}

        sampled_diagnostics = {
            "S8193": dense_check(8193, False),
            "event_aware_S32768": dense_check(32768, True),
        }

        retained_fields = {
            "timing_controls": retained.timing_controls,
            "path_parameter": retained.path_parameter,
            "time": retained.time,
            "position": retained.joint_state.position,
            "velocity": retained.joint_state.velocity,
            "acceleration": retained.joint_state.acceleration,
            "jerk": retained.joint_state.jerk,
            "success": retained.success,
            "status": retained.status,
            "optimization_ok": retained.optimization_ok,
            "uniform_duration": retained.uniform_duration,
            "stationary": retained.stationary,
            "effective_geometry": trajectory.control_points,
            "effective_timing_controls": trajectory.timing.controls,
            "effective_time_grid": trajectory.timing.time,
        }
        retained_digests_before = {name: _digest(value) for name, value in retained_fields.items()}
        solver.timeopt((controls * 0.9).contiguous(), start, goal * 0.9)
        torch.cuda.synchronize(device)
        retained_digests_after = {name: _digest(value) for name, value in retained_fields.items()}
        if retained_digests_before != retained_digests_after:
            failed.append("borrowed_output_clone_lifetime")
        result = {
            "evidence_kind": "synthetic_native_adapter_parity_smoke",
            "upstream_sha": UPSTREAM_SHA,
            "native_distribution": "nvidia-curobo",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "seed": 20261110,
            "trajectory": "one-joint linear-action-control smoke path; not a Panda/UR5 cohort member",
            "source_id": "synthetic-native-smoke-seed-20261110",
            "scene_id": "empty synthetic pendulum scene; no collision query",
            "model": {"name": "one-link pendulum", "payload": "none",
                      "urdf_sha256": hashlib.sha256(urdf.encode()).hexdigest()},
            "timeopt_config": {
                "batch_size": 1, "n_knots": 16, "timing_controls": 32,
                "num_intervals": 511, "max_iterations": 10,
                "limit_scale": 0.95, "jerk_limit_scale": 0.95,
                "use_cuda_graph": True, "enable_inverse_dynamics": False,
                "physical_limits": {"velocity": [2.0], "acceleration": [3.0], "jerk": [50.0]},
                "planning_scaled_limits": {"velocity": [1.9], "acceleration": [2.85], "jerk": [47.5]},
            },
            "timing_controls_count": int(trajectory.timing.controls.shape[-1]),
            "duration_seconds": duration,
            "path_spans": spans,
            "timing_knots": len(solver._basis.breaks_cpu),
            "union_interior_knots": len(knots),
            "random_interior_points": 1001,
            "knot_side_points": len(sides),
            "total_points": len(progress),
            "tolerance": tolerance,
            "errors": errors,
            "dense_diagnostics": sampled_diagnostics,
            "input_hashes": {
                "native_controls": _digest(controls),
                "requested_start": _digest(start),
                "requested_goal": _digest(goal),
                "effective_geometry": _digest(trajectory.control_points),
                "timing_coefficients": _digest(trajectory.timing.controls),
                "timing_basis_polynomial": _digest(solver._basis.polynomial),
            },
            "frozen_arrays": {"path": str(args.arrays_out), "sha256": arrays_sha256},
            "borrowed_result_clone": "PASS" if retained_digests_before == retained_digests_after else "FAIL",
            "borrowed_clone_hashes_before": retained_digests_before,
            "borrowed_clone_hashes_after": retained_digests_after,
            "status": "PASS" if not failed else "FAIL",
            "failed_fields": failed,
            "limitations": [
                "synthetic one-joint native smoke only; no Panda/UR5 source-query path",
                "consistency tolerance validates adapter parity, not continuous-time feasibility",
                "native FP32 kernel values compared with float64 evaluation of serialized float32 controls",
            ],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
