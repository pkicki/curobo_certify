#!/usr/bin/env python3
"""Plan C-space queries and retain native cubic TimeOpt outputs.

cuRobo's MotionPlanner integrates its pinned TrajectoryTimeOpt with the
optimized cubic B-spline. The returned spline_trajectory contains the actual
effective geometry and 32 timing coefficients; no sampled-path fitting is
performed here.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curobo_certify.adapters.timeopt import UPSTREAM_SHA, sha256_array  # noqa: E402
from curobo_certify.query_pipeline import pilot_queries, verify_query  # noqa: E402


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_polynomial(points: np.ndarray, s: float) -> tuple[np.ndarray, ...]:
    """Evaluate cuRobo's uniform cardinal cubic path and s-derivatives."""
    spans = points.shape[0] - 3
    z = min(max(float(s), 0.0), 1.0) * spans
    segment = min(int(z), spans - 1)
    u = z - segment
    support = points[segment:segment + 4].astype(np.float64)
    matrix6 = np.asarray([
        [-1, 3, -3, 1], [3, -6, 0, 4], [-3, 3, 3, 1], [1, 0, 0, 0]
    ], dtype=np.float64) / 6.0
    descending = support.T @ matrix6
    c0, c1, c2, c3 = (descending[:, i] for i in (3, 2, 1, 0))
    q = ((c3 * u + c2) * u + c1) * u + c0
    qs = ((3 * c3 * u + 2 * c2) * u + c1) * spans
    qss = (6 * c3 * u + 2 * c2) * spans**2
    qsss = 6 * c3 * spans**3
    return q, qs, qss, qsss


def _native_timing_state(opt, controls: np.ndarray, requested: np.float32):
    """Evaluate the pinned timing polynomial and five-step inversion in FP32.

    This mirrors the public equation contract, including its fixed GL6 cell
    quadrature. It is used only to compare adapter evaluation with native GPU
    samples; the proof checker consumes exact rational coefficients separately.
    """
    polynomial = opt._basis.polynomial.detach().cpu().numpy()
    time = opt._times[0].detach().cpu().numpy()
    nodes = opt._nodes.detach().cpu().numpy()
    weights = opt._weights.detach().cpu().numpy()
    zero, one = np.float32(0), np.float32(1)

    def state(s: np.float32):
        span = min(28, max(0, int(np.float32(s * np.float32(29)))))
        left = np.float32(span) / np.float32(29)
        local = np.float32(s - left)
        a = np.zeros(4, dtype=np.float32)
        for power in range(4):
            for k in range(4):
                product = np.float32(polynomial[span, power, span + k] * controls[span + k])
                a[power] = np.float32(a[power] + product)
        x = np.float32(np.float32(np.float32(a[0] * local + a[1]) * local + a[2]) * local + a[3])
        x1 = np.float32(np.float32(np.float32(3) * a[0] * local + np.float32(2) * a[1]) * local + a[2])
        x2 = np.float32(np.float32(6) * a[0] * local + np.float32(2) * a[1])
        return x, x1, x2

    cell = min(len(time) - 2, max(0, int(np.searchsorted(time, requested, side="right") - 1)))
    left = np.float32(cell) / np.float32(len(time) - 1)
    right = np.float32(cell + 1) / np.float32(len(time) - 1)
    denom = np.float32(max(float(np.float32(time[cell + 1] - time[cell])), 1e-20))
    s = np.float32(left + np.float32(right - left) * np.float32(requested - time[cell]) / denom)
    for _ in range(5):
        elapsed = zero
        for k in range(6):
            sample_s = np.float32(left + np.float32(s - left) * nodes[k])
            x, _, _ = state(sample_s)
            elapsed = np.float32(elapsed + np.float32(1.0 / np.sqrt(max(float(x), 1e-20))) * weights[k])
        x, x1, x2 = state(s)
        residual = np.float32(np.float32(time[cell] + np.float32(s - left) * elapsed) - requested)
        candidate = np.float32(s - np.float32(residual * np.sqrt(max(float(x), 1e-20))))
        s = np.float32(max(float(left), min(float(right), float(candidate))))
    s = zero if requested <= 0 else one if requested >= time[-1] else s
    return float(s), tuple(float(v) for v in state(s))


def _parity(opt, points: np.ndarray, timing: np.ndarray, duration: float, seed: int):
    """Compare native GPU samples to independent spline equations."""
    rng = np.random.default_rng(seed)
    spline = opt._basis.spline
    x_spline = BSpline(spline.t, timing, spline.k, extrapolate=False)
    knots = sorted(set([i / (points.shape[0] - 3) for i in range(1, points.shape[0] - 3)]
                       + list(opt._basis.breaks_cpu[1:-1])))

    def time_at_progress(s: float) -> float:
        if s <= 0:
            return 0.0
        return quad(lambda z: 1.0 / math.sqrt(float(x_spline(z))), 0.0, s,
                    epsabs=1e-10, epsrel=1e-10, limit=200)[0]

    random_t = rng.uniform(max(1e-7 * duration, 1e-8), duration * (1 - 1e-7), 1001)
    side_t = []
    for knot in knots:
        # 1e-4 progress separates the two sides in native FP32 time samples;
        # smaller offsets can quantize to the knot and silently test one side.
        epsilon = min(1e-4, knot / 4, (1 - knot) / 4)
        for progress in (knot - epsilon, knot + epsilon):
            value = time_at_progress(progress)
            if 0 < value < duration:
                side_t.append(value)
    times = np.concatenate((random_t, np.asarray(side_t)))
    import torch
    sample_time = torch.as_tensor(times, dtype=torch.float32, device=opt.device)[None].contiguous()
    values, progress = opt._sample(sample_time)
    torch.cuda.synchronize(opt.device)
    native = values.detach().cpu().numpy().astype(np.float64)[:, 0]
    reported_progress = progress.detach().cpu().numpy().astype(np.float64)[0]
    # The sampler reports an offset-adjusted normalized coordinate. Reconstruct
    # the full effective-spline coordinate used by the exported control net.
    start_offset = int(not opt._reference)
    spans = int(points.shape[0] - 3)
    progress = (reported_progress * (opt.config.n_knots + 2) - start_offset + 1) / spans
    # The public progress output is clamped to [0,1]. For reference-boundary
    # mode it loses coordinate information throughout the first boundary span.
    # Recover that span from the solver's retained physical-time integration
    # table (uniform in complete-spline coordinate).
    clamped = (reported_progress <= 0.0) | (reported_progress >= 1.0)
    if np.any(clamped):
        time_grid = opt._times[0].detach().cpu().numpy().astype(np.float64)
        progress_grid = np.linspace(0.0, 1.0, len(time_grid))
        progress[clamped] = np.interp(times[clamped], time_grid, progress_grid)
    native_timing = [_native_timing_state(opt, timing, np.float32(t)) for t in times]
    progress = np.asarray([item[0] for item in native_timing], dtype=np.float64)
    x, x1, x2 = (np.asarray([item[1][index] for item in native_timing], dtype=np.float64)
                 for index in range(3))
    geometry = np.asarray([_path_polynomial(points, s) for s in progress])
    q, qs, qss, qsss = (geometry[:, i, :] for i in range(4))
    speed = np.sqrt(np.maximum(x, 0))[:, None]
    v = qs * speed
    a = qss * x[:, None] + 0.5 * qs * x1[:, None]
    jerk = speed * (qsss * x[:, None] + 1.5 * qss * x1[:, None] + 0.5 * qs * x2[:, None])
    tiny = np.finfo(np.float64).tiny
    recovered_qs = native[1] / np.maximum(speed, tiny)
    recovered_qss = (native[2] - 0.5 * recovered_qs * x1[:, None]) / np.maximum(x[:, None], tiny)
    recovered_qsss = (native[3] / np.maximum(speed, tiny) - 1.5 * recovered_qss * x1[:, None]
                      - 0.5 * recovered_qs * x2[:, None]) / np.maximum(x[:, None], tiny)
    # Physical samples cannot stably identify geometric derivatives at a
    # zero-speed boundary. Audit q_s/q_ss/q_sss there against cuRobo's direct
    # native path workspace below; retain physical q/v/a/j parity everywhere.
    observable = x > 1e-3
    compare = {"position": (native[0], q), "velocity": (native[1], v), "acceleration": (native[2], a),
               "jerk": (native[3], jerk)}
    errors = {}
    for name, (actual, expected) in compare.items():
        delta = np.abs(actual - expected)
        errors[name] = {"max_abs": float(delta.max()),
                        "max_scaled_rel": float((delta / np.maximum(1.0, np.abs(expected))).max())}
    grid = opt._output_grid_cpu
    native_path = opt._path[0, 0, :len(grid)].detach().cpu().numpy().astype(np.float64)
    native_path_all = opt._path[:, 0, :len(grid)].detach().cpu().numpy().astype(np.float64)
    reference_path_all = np.asarray([_path_polynomial(points, s) for s in grid]).transpose(1, 0, 2)
    for derivative, name in enumerate(("q", "q_s", "q_ss", "q_sss")):
        delta = np.abs(native_path_all[derivative] - reference_path_all[derivative])
        errors[f"native_internal_{name}_grid"] = {
            "max_abs": float(delta.max()),
            "max_scaled_rel": float((delta / np.maximum(1.0, np.abs(reference_path_all[derivative]))).max())}
    for name, actual, expected in (("path_q_s", recovered_qs, qs),
                                   ("path_q_ss", recovered_qss, qss),
                                   ("path_q_sss", recovered_qsss, qsss)):
        if np.any(observable):
            delta = np.abs(actual[observable] - expected[observable])
            errors[name] = {"max_abs": float(delta.max()),
                            "max_scaled_rel": float((delta / np.maximum(1.0, np.abs(expected[observable]))).max()),
                            "observable_samples": int(observable.sum()), "x_threshold": 1e-3}
    tolerance = 4e-4
    physical_limits = []
    planning_limits = []
    for family_index, family in enumerate(("velocity", "acceleration", "jerk")):
        physical_limits.append(np.asarray(opt.config.robot.get_joint_limits().__getattribute__(family)
                                          .detach().cpu().numpy()))
        planning_limits.append(opt._limits[family_index].detach().cpu().numpy())

    def dense_diagnostic(count: int, event_aware: bool):
        dense_t = np.linspace(0.0, duration, count, dtype=np.float64)
        events = []
        if event_aware:
            occupied: set[int] = set()
            path_knots = {i / (points.shape[0] - 3)
                          for i in range(1, points.shape[0] - 3)}
            timing_knots = set(float(x) for x in opt._basis.breaks_cpu[1:-1])
            for knot in knots:
                epsilon = min(1e-4, knot / 4, (1 - knot) / 4)
                for side, s_event in (("left", knot - epsilon), ("right", knot + epsilon)):
                    t_event = time_at_progress(s_event)
                    target = int(round(t_event / duration * (count - 1)))
                    slot = max(0, min(count - 1, target))
                    while slot in occupied and slot + 1 < count:
                        slot += 1
                    occupied.add(slot)
                    dense_t[slot] = t_event
                    events.append({"knot": float(knot), "side": side, "time": float(t_event),
                                   "index": slot, "requested_progress": float(s_event),
                                   "family": ("both" if any(abs(knot - x) < 1e-12 for x in path_knots)
                                              and any(abs(knot - x) < 1e-12 for x in timing_knots)
                                              else "path" if any(abs(knot - x) < 1e-12 for x in path_knots)
                                              else "timing")})
            dense_t.sort()
            for event in events:
                event["index"] = int(np.argmin(np.abs(dense_t - event["time"])))
        tensor_times = torch.as_tensor(dense_t.astype(np.float32), device=opt.device)[None].contiguous()
        dense_values, dense_progress = opt._sample(tensor_times)
        torch.cuda.synchronize(opt.device)
        states = dense_values.detach().cpu().numpy()[:, 0].copy()
        progress_values = dense_progress.detach().cpu().numpy()[0].copy()
        observed = 0
        if event_aware:
            for event in events:
                requested_progress = event["requested_progress"]
                actual_progress = _native_timing_state(opt, timing, np.float32(event["time"]))[0]
                event["sampled_progress"] = actual_progress
                event["requested_progress"] = requested_progress
                event["side_verified"] = (actual_progress < event["knot"] if event["side"] == "left"
                                           else actual_progress > event["knot"])
                observed += int(event["side_verified"])
        ratios = {}
        for family_index, family in enumerate(("velocity", "acceleration", "jerk")):
            absolute = np.maximum(np.abs(physical_limits[family_index][0]),
                                  np.abs(physical_limits[family_index][1]))
            ratios[family] = {
                "physical_max_ratio": float(np.max(np.abs(states[family_index + 1]) / absolute)),
                "planning_max_ratio": float(np.max(np.abs(states[family_index + 1]) / planning_limits[family_index])),
            }
        return {"sample_count": count, "grid": "event_aware" if event_aware else "uniform_time",
                "ratios": ratios, "events": events,
                "observed_event_count": observed, "expected_event_count": 2 * len(knots),
                "times": dense_t.astype(np.float32), "progress": progress_values,
                "states": states.astype(np.float32)}

    diagnostic_8193 = dense_diagnostic(8193, False)
    diagnostic_32768 = dense_diagnostic(32768, True)
    return {"status": "PASS" if all(v["max_abs"] <= tolerance and
                                      v["max_scaled_rel"] <= tolerance for v in errors.values())
            else "FAIL", "tolerance": tolerance, "random_interior_points": 1001,
            "one_sided_knot_points": len(side_t), "common_knots": len(knots),
            "errors": errors,
            "native_reported_progress_transform": "s_complete=(s_reported*(n_knots+2)-start_offset+1)/(n_knots+3-start_offset)",
            "derivative_recovery": {"x_threshold": 1e-3, "observable_samples": int(observable.sum()),
                                    "total_samples": len(progress),
                                    "endpoint_derivatives_checked_against_native_path_workspace": True},
            "dense_diagnostics": [{key: value for key, value in diagnostic.items()
                                   if key not in ("times", "progress", "states", "events")}
                                  | {"events": diagnostic["events"]}
                                  for diagnostic in (diagnostic_8193, diagnostic_32768)],
            "samples": {"times": times, "progress": progress,
                        "reported_progress": reported_progress, "native": native,
                        "reference": np.asarray([q, qs, qss, qsss, v, a, jerk]),
                        "diagnostic_8193": diagnostic_8193,
                        "diagnostic_32768": diagnostic_32768}}


def _install_cuda_property_compatibility() -> dict:
    """Restore launch-sizing attributes missing from this PyTorch build.

    cuRobo c301cb reads shared_memory_per_block(_optin) while torch 2.5.1's
    Python property object omits those attributes. Values come from the loaded
    CUDA runtime API (cudaDeviceGetAttribute), never from guessed constants.
    Only the metadata accessor is wrapped; planning/TimeOpt code is untouched.
    """
    import torch
    original = torch.cuda.get_device_properties
    runtime = ctypes.CDLL("libcudart.so.12")
    get_attribute = runtime.cudaDeviceGetAttribute
    get_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    get_attribute.restype = ctypes.c_int

    def attribute(attr: int, device: int) -> int:
        result = ctypes.c_int()
        error = get_attribute(ctypes.byref(result), attr, device)
        if error:
            raise RuntimeError(f"cudaDeviceGetAttribute({attr}) failed with {error}")
        return int(result.value)

    class PropertiesProxy:
        def __init__(self, wrapped, device: int):
            self._wrapped = wrapped
            self.shared_memory_per_block = attribute(8, device)
            self.shared_memory_per_block_optin = attribute(97, device)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def compatible(device=None):
        logical = torch.cuda.current_device() if device is None else torch.cuda._get_device_index(device)
        return PropertiesProxy(original(device), logical)

    torch.cuda.get_device_properties = compatible
    props = compatible(torch.cuda.current_device())
    return {"shim": "torch_cuda_device_property_launch_metadata",
            "shared_memory_per_block": props.shared_memory_per_block,
            "shared_memory_per_block_optin": props.shared_memory_per_block_optin,
            "device": props.name}


def _dump_failure(path: Path, row: dict, stage: str, reason: str, **extra) -> None:
    value = {"source_id": row["source_id"], "source_query_sha256": row["source_query_sha256"],
             "robot": row["robot"], "scene_id": row["scene_id"], "stage": stage,
             "status": "FAILED_RETAINED", "reason": reason, **extra}
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _jsonable(value):
    """Convert nested diagnostic metadata without retaining borrowed tensors."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _run_row(row: dict, output_dir: Path, robot_configs: dict[str, str]) -> dict:
    if not verify_query(row):
        raise ValueError(f"source query hash mismatch: {row.get('source_id')}")
    if row["robot"] not in robot_configs:
        raise ValueError(f"no verified native robot configuration for {row['robot']}")
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    import torch

    robot_cfg = robot_configs[row["robot"]]
    scene_cfg = None if row["scene_id"] == "empty" else row["scene_id"] + ".yml"
    from curobo.content import get_robot_configs_path, get_scene_configs_path
    robot_path = Path(get_robot_configs_path()) / robot_cfg
    scene_path = Path(get_scene_configs_path()) / scene_cfg if scene_cfg else None
    planner_settings = {
        "graph_planner_algorithm": "rrt", "enable_trajopt": True,
        "enable_timeopt": True, "enable_inverse_dynamics": True,
        "use_cuda_graph": True, "num_ik_seeds": 4, "num_trajopt_seeds": 1,
        "max_attempts": 2, "enable_graph_attempt": 0, "finetune_attempts": 0,
    }
    config = MotionPlannerCfg.create(
        robot=robot_cfg,
        scene_model=scene_cfg,
        graph_planner_algorithm="rrt",
        enable_trajopt=True,
        enable_timeopt=True,
        enable_inverse_dynamics=True,
        use_cuda_graph=True,
        num_ik_seeds=4,
        num_trajopt_seeds=1,
        random_seed=int(row["seed"] + row["query_index"]),
    )
    started = time.perf_counter()
    with MotionPlanner(config) as planner:
        default = planner.default_joint_state.clone()
        start = default.position.reshape(1, -1)
        active_dof = start.shape[-1]
        if len(row["goal_delta_rad"]) != active_dof:
            raise ValueError(f"model dof {active_dof} does not match query dof {len(row['goal_delta_rad'])}")
        frozen_start = row.get("start_joint_positions_rad")
        frozen_goal = row.get("goal_joint_positions_rad")
        if frozen_start is None or frozen_goal is None:
            raise ValueError("query lacks frozen absolute start/goal positions")
        frozen_start_tensor = torch.tensor(frozen_start, device=start.device, dtype=start.dtype)[None]
        if not torch.equal(start, frozen_start_tensor):
            raise ValueError("planner default state differs from frozen source-query start")
        start = frozen_start_tensor
        goal = torch.tensor(frozen_goal, device=start.device, dtype=start.dtype)[None]
        current = JointState.from_position(start, joint_names=planner.joint_names)
        target = JointState.from_position(goal, joint_names=planner.joint_names)
        result = planner.plan_cspace(target, current, max_attempts=2, enable_graph_attempt=0,
                                     finetune_attempts=0)
        torch.cuda.synchronize(start.device)
        solve_wall = time.perf_counter() - started
        if result is None:
            raise RuntimeError("planner returned no result")
        retained = result.clone()
        succeeded = bool(torch.as_tensor(retained.success).any().item())
        if not succeeded:
            raise RuntimeError(f"native planner/TimeOpt failure: success={retained.success.tolist()}, "
                               f"status={getattr(retained, 'timeopt_success', None)}")
        spline = retained.spline_trajectory
        if spline is None or not hasattr(spline.timing, "controls"):
            raise RuntimeError("successful result omitted effective TimeOpt spline/timing controls")
        points = spline.control_points.detach().cpu().numpy().copy()
        timing = spline.timing.controls.detach().cpu().numpy().copy()
        duration = float(spline.duration.reshape(-1)[0].item())
        if points.shape[0] != 1 or timing.shape != (1, 32):
            raise RuntimeError(f"unexpected exported layouts: geometry {points.shape}, timing {timing.shape}")
        model_limits = planner.kinematics.config.get_joint_limits()
        limits = {}
        for family in ("velocity", "acceleration", "jerk"):
            bounds = getattr(model_limits, family, None)
            if bounds is None:
                limits[family] = None
            else:
                limits[family] = bounds.detach().cpu().numpy().copy()
        native_optimizer = planner.trajopt_solver._timeopt.optimizer
        parity = _parity(native_optimizer, points[0], timing[0], duration,
                         int(row["seed"] + row["query_index"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = output_dir / f"{row['source_id']}.npz"
        arrays = {
            "effective_geometry": points[0], "timing_controls": timing[0],
            "requested_start": start.detach().cpu().numpy(),
            "requested_goal": goal.detach().cpu().numpy(),
            "native_duration": np.asarray([duration], dtype=np.float64),
            "native_success": np.asarray([succeeded]),
            "parity_times": parity["samples"]["times"],
            "parity_progress": parity["samples"]["progress"],
            "parity_native": parity["samples"]["native"],
            "parity_reference": parity["samples"]["reference"],
            "timing_knot_vector": native_optimizer._basis.spline.t,
            "native_timing_time_grid": native_optimizer._times[0].detach().cpu().numpy().copy(),
            "native_timing_status": native_optimizer._result.status.detach().cpu().numpy().copy(),
            "native_timing_optimization_ok": native_optimizer._result.optimization_ok.detach().cpu().numpy().copy(),
            "native_timing_uniform_duration": native_optimizer._result.uniform_duration.detach().cpu().numpy().copy(),
            "diagnostic_8193_times": parity["samples"]["diagnostic_8193"]["times"],
            "diagnostic_8193_states": parity["samples"]["diagnostic_8193"]["states"],
            "diagnostic_8193_progress": parity["samples"]["diagnostic_8193"]["progress"],
            "diagnostic_32768_times": parity["samples"]["diagnostic_32768"]["times"],
            "diagnostic_32768_states": parity["samples"]["diagnostic_32768"]["states"],
            "diagnostic_32768_progress": parity["samples"]["diagnostic_32768"]["progress"],
        }
        for family, value in limits.items():
            if value is not None:
                arrays[f"{family}_limits"] = value
        np.savez_compressed(arrays_path, **arrays)
        array_sha = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        record = {
            "evidence_kind": "real_native_planner_timeopt_pilot",
            "campaign_data": True,
            "source_id": row["source_id"], "source_query": row,
            "source_query_sha256": row["source_query_sha256"],
            "robot_config": robot_cfg, "scene_config": scene_cfg,
            "robot_config_sha256": _file_sha256(robot_path),
            "scene_config_sha256": _file_sha256(scene_path) if scene_path else None,
            "planner_settings": planner_settings,
            "upstream_sha": UPSTREAM_SHA, "planner_source": retained.trajectory_source,
            "planner_success": succeeded,
            "planner_validation": _jsonable(retained.debug_info.get("validation")),
            "native_timeopt_duration_seconds": retained.timeopt_duration.detach().cpu().tolist()
                if retained.timeopt_duration is not None else None,
            "native_duration_seconds": duration,
            "native_adapter_parity": {key: value for key, value in parity.items() if key != "samples"},
            "effective_geometry_shape": list(points[0].shape),
            "timing_controls_shape": list(timing[0].shape),
            "geometry_hash": sha256_array(points), "timing_hash": sha256_array(timing),
            "arrays_path": str(arrays_path), "arrays_sha256": array_sha,
            "limits_available": {key: value is not None for key, value in limits.items()},
            "planner_complete_request_wall_seconds": solve_wall,
            "conversion": "none: cloned selected effective cubic spline produced by pinned planner-integrated native TimeOpt",
            "native_torque_constraint": "ENABLED_BY_PINNED_TIMEOPT_CONFIGURATION",
            "full_dynamics_certificate": "UNSUPPORTED_DYNAMICS",
            "status": "NATIVE_SUCCESS_PARITY_PASS" if parity["status"] == "PASS"
                else "NATIVE_SUCCESS_PARITY_FAIL",
        }
        return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path("runs/native_pilot_queries.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/native_pilot"))
    parser.add_argument("--robot-config", action="append", default=[], metavar="ROBOT=CONFIG")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--robot", choices=("panda", "ur5e"))
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("native query generation requires CUDA")
    compat = _install_cuda_property_compatibility()
    if args.queries.exists():
        rows = json.loads(args.queries.read_text())
    else:
        rows = pilot_queries()
    configs = {"panda": "franka.yml"}
    for item in args.robot_config:
        key, value = item.split("=", 1)
        configs[key] = value
    if args.robot:
        rows = [row for row in rows if row["robot"] == args.robot]
    rows = rows[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        record_path = args.out_dir / f"{row['source_id']}.json"
        if args.skip_existing and record_path.exists():
            existing = json.loads(record_path.read_text())
            if existing.get("source_query_sha256") == row.get("source_query_sha256"):
                records.append(existing)
                print(f"{row['source_id']}: retained existing {existing.get('status')}")
                continue
        try:
            record = _run_row(row, args.out_dir, configs)
            record["cuda_compatibility"] = compat
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
            records.append(record)
            print(f"{row['source_id']}: {record['status']} duration={record['native_duration_seconds']:.9g}s")
        except Exception as exc:
            _dump_failure(record_path, row, "planner_or_timeopt", f"{type(exc).__name__}: {exc}",
                          cuda_compatibility=compat)
            records.append(json.loads(record_path.read_text()))
            print(f"{row['source_id']}: FAILED_RETAINED {type(exc).__name__}: {exc}")
    summary = {"evidence_kind": "native_source_query_pilot_summary", "campaign_data": True,
               "query_manifest": str(args.queries),
               "query_manifest_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest()
                   if args.queries.exists() else None,
               "upstream_sha": UPSTREAM_SHA, "cuda_compatibility": compat,
               "queries_attempted": len(rows),
               "native_successes": sum(item.get("status") in ("NATIVE_SUCCESS_PARITY_PASS", "NATIVE_SUCCESS_PARITY_FAIL") for item in records),
               "adapter_parity_passes": sum(item.get("status") == "NATIVE_SUCCESS_PARITY_PASS" for item in records),
               "adapter_parity_failures": sum(item.get("status") == "NATIVE_SUCCESS_PARITY_FAIL" for item in records),
               "retained_failures": sum(item.get("status") == "FAILED_RETAINED" for item in records),
               "records": [str(args.out_dir / f"{row['source_id']}.json") for row in rows]}
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
