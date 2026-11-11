#!/usr/bin/env python3
"""Join intrusive CUDA operand traces to the eight immutable Panda pilot paths."""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from diagnose_native_jerk import path_fp32, timing_replay, fadd, fmul  # noqa: E402


def bits(value):
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def analyze(trace_path: Path, pilot_dir: Path, out_dir: Path, instrumented_dir: Path | None = None):
    traces = json.loads(trace_path.read_text())['records']
    out_dir.mkdir(parents=True, exist_ok=True)
    fmaf = ctypes.CDLL('libm.so.6').fmaf
    fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fmaf.restype = ctypes.c_float
    rows = []
    source_models = {}
    for metadata_path in sorted(pilot_dir.glob('pilot-panda-*.json')):
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get('planner_success'):
            continue
        array_path = ROOT / metadata['arrays_path']
        if __import__('hashlib').sha256(array_path.read_bytes()).hexdigest() != metadata['arrays_sha256']:
            raise ValueError(f"frozen source SHA mismatch: {array_path}")
        arrays = np.load(array_path, allow_pickle=False)
        source_models[metadata['source_id']] = (metadata, arrays)

    for trace in traces:
        idx, joint = trace['idx'], trace['joint']
        if idx not in set(range(50, 1001, 50)) | {1035}:
            continue
        target_time_bits = trace.get('t_bits')
        candidates = []
        for source_id, (metadata, arrays) in source_models.items():
            if idx >= len(arrays['parity_times']):
                continue
            if f'{bits(arrays["parity_times"][idx]):08x}' != target_time_bits:
                continue
            geom = arrays['effective_geometry'].copy()
            timing = arrays['timing_controls'].astype(np.float32)
            time_grid = arrays['native_timing_time_grid'].astype(np.float32)
            t = np.float32(trace['t'])
            s, timing_values, _ = timing_replay(timing, time_grid, arrays['native_duration'][0], t)
            q_values = path_fp32(geom, s)
            if int(q_values[4]) != trace['seg']:
                continue
            score = abs(s - trace['s']) + sum(abs(float(timing_values[k]) - trace[f'x{k}'])
                                               for k in range(3))
            score += sum(abs(float(q_values[k][joint]) - trace[f'q{k}']) for k in range(4))
            candidates.append((score, source_id, metadata, arrays, s, timing_values, q_values))
        if not candidates:
            continue
        _, source_id, metadata, arrays, s_replay, x_replay, q_replay = min(candidates, key=lambda x: x[0])
        native = np.float32(arrays['parity_native'][3, idx, joint])
        adapter = float(arrays['parity_reference'][6, idx, joint])
        q1, q2, q3 = (np.float32(q_replay[k][joint]) for k in (1, 2, 3))
        x0, x1, x2 = (np.float32(x) for x in x_replay)
        speed = np.float32(np.sqrt(max(float(x0), 0.0)))
        t1 = fmul(q3, x0); t2 = fmul(fmul(np.float32(1.5), q2), x1)
        t3 = fmul(fmul(np.float32(0.5), q1), x2)
        replay = fmul(speed, fadd(fadd(t1, t2), t3))
        m2, m3 = fmul(np.float32(1.5), q2), fmul(np.float32(0.5), q1)
        fma_sum = np.float32(fmaf(float(m2), float(x1), float(fmul(q3, x0))))
        fma_sum = np.float32(fmaf(float(m3), float(x2), float(fma_sum)))
        replay_fma = fmul(speed, fma_sum)
        dq1, dq2, dq3 = (np.float32(trace[f'q{k}']) for k in (1, 2, 3))
        dx0, dx1, dx2 = (np.float32(trace[f'x{k}']) for k in range(3))
        dspeed = np.float32(trace['speed'])
        d_m2, d_m3 = fmul(np.float32(1.5), dq2), fmul(np.float32(0.5), dq1)
        device_fma_sum = np.float32(fmaf(float(d_m2), float(dx1), float(fmul(dq3, dx0))))
        device_fma_sum = np.float32(fmaf(float(d_m3), float(dx2), float(device_fma_sum)))
        device_operand_fma_jerk = fmul(dspeed, device_fma_sum)
        progress_coordinate = trace['s'] * (arrays['effective_geometry'].shape[0] - 3)
        knot_distance = abs(trace['s'] * (arrays['effective_geometry'].shape[0] - 3)
                            - round(trace['s'] * (arrays['effective_geometry'].shape[0] - 3)))
        rows.append({
            'path_id': source_id, 'source_arrays_sha256': metadata['arrays_sha256'],
            'trace_idx': idx, 'joint': joint, 'physical_time_s': trace['t'],
            'internal_progress_device': trace['s'], 'internal_progress_replay': s_replay,
            'progress_error': s_replay - trace['s'], 'segment_device': trace['seg'],
            'segment_replay': int(q_replay[4]), 'local_device': trace['local'],
            'local_replay': float(q_replay[5]), 'distance_to_geometric_knot_u': knot_distance,
            'x_device': [trace[f'x{k}'] for k in range(3)], 'x_replay_fp32': list(x_replay),
            'q_derivatives_device': [trace[f'q{k}'] for k in (1, 2, 3)],
            'q_derivatives_replay_fp32': [float(q_replay[k][joint]) for k in (1, 2, 3)],
            'jerk_native_pristine': float(native), 'jerk_device_instrumented': trace['native_jerk'],
            'jerk_adapter_fp64': adapter, 'jerk_source_replay_no_fma': float(replay),
            'jerk_source_replay_fma_from_replayed_operands': float(replay_fma),
            'jerk_device_operand_fma_replay': float(device_operand_fma_jerk),
            'native_vs_adapter_abs': abs(float(native) - adapter),
            'native_vs_no_fma_replay_abs': abs(float(native) - float(replay)),
            'native_vs_fma_replay_abs': abs(float(native) - float(replay_fma)),
            'native_matches_fma_replay': bits(native) == bits(replay_fma),
            'native_matches_device_operand_fma': bits(np.float32(trace['native_jerk'])) == bits(device_operand_fma_jerk),
            'diagnostic_terms_device_recomputed': [trace.get(f't{k}') for k in (1, 2, 3)],
        })
    summaries = {}
    for path_id in sorted({r['path_id'] for r in rows}):
        selected = [r for r in rows if r['path_id'] == path_id]
        summaries[path_id] = {
            'matched_records': len(selected), 'distinct_times': len({r['trace_idx'] for r in selected}),
            'joints': sorted({r['joint'] for r in selected}),
            'max_abs_progress_error': max(abs(r['progress_error']) for r in selected),
            'max_abs_x_errors': [max(abs(r['x_device'][k] - r['x_replay_fp32'][k]) for r in selected) for k in range(3)],
            'max_abs_q_derivative_errors': [max(abs(r['q_derivatives_device'][k] - r['q_derivatives_replay_fp32'][k]) for r in selected) for k in range(3)],
            'max_native_vs_no_fma_replay_abs': max(r['native_vs_no_fma_replay_abs'] for r in selected),
            'max_native_vs_fma_replay_abs': max(r['native_vs_fma_replay_abs'] for r in selected),
            'native_matches_fma_replay_count': sum(r['native_matches_fma_replay'] for r in selected),
            'native_matches_device_operand_fma_count': sum(r['native_matches_device_operand_fma'] for r in selected),
        }
    instrumentation_comparison = {}
    if instrumented_dir is not None:
        for path_id, (_, source_arrays) in source_models.items():
            instrumented_path = instrumented_dir / f'{path_id}.npz'
            with np.load(instrumented_path, allow_pickle=False) as instrumented:
                fields = ('effective_geometry', 'timing_controls', 'native_timing_time_grid', 'parity_times',
                          'parity_native', 'diagnostic_8193_times', 'diagnostic_8193_states',
                          'diagnostic_32768_times', 'diagnostic_32768_states')
                same = {name: bool(np.array_equal(source_arrays[name], instrumented[name])) for name in fields}
            instrumentation_comparison[path_id] = {'inputs_and_outputs_bitwise_equal': all(same.values()),
                                                   'fieldwise_bitwise_equal': same,
                                                   'instrumented_npz_sha256': __import__('hashlib').sha256(instrumented_path.read_bytes()).hexdigest()}
    report = {'scope': 'INTRUSIVE_DEVICE_TRACE_DIAGNOSTIC_NOT_CERTIFICATION',
              'registered_parity_tolerance_scaled_relative': 4e-4,
              'trace_sha256': __import__('hashlib').sha256(trace_path.read_bytes()).hexdigest(),
              'matched_trace_rows': len(rows), 'path_summaries': summaries,
              'pristine_instrumented_comparison': instrumentation_comparison, 'records': rows}
    (out_dir / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    for path_id in summaries:
        data = [r for r in rows if r['path_id'] == path_id]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].scatter([r['physical_time_s'] for r in data], [r['native_vs_adapter_abs'] for r in data], c=[r['joint'] for r in data], cmap='viridis')
        axes[0,0].set(xlabel='physical time [s]', ylabel='|native jerk − adapter|')
        axes[0,1].scatter([r['internal_progress_device'] for r in data], [r['native_vs_no_fma_replay_abs'] for r in data], c=[r['joint'] for r in data], cmap='viridis')
        axes[0,1].set(xlabel='internal progress s', ylabel='|native − no-FMA replay|')
        axes[1,0].scatter([r['joint'] for r in data], [r['native_vs_adapter_abs'] for r in data], c=[r['distance_to_geometric_knot_u'] for r in data], cmap='plasma')
        axes[1,0].set(xlabel='joint', ylabel='|native jerk − adapter|')
        axes[1,1].scatter([r['distance_to_geometric_knot_u'] for r in data], [r['native_vs_no_fma_replay_abs'] for r in data], c=[r['joint'] for r in data], cmap='viridis')
        axes[1,1].set(xlabel='distance to geometric knot in span units', ylabel='|native − no-FMA replay|')
        for ax in axes.flat: ax.grid(alpha=.2)
        fig.suptitle(f'{path_id}: CUDA trace is intrusive and diagnostic only')
        fig.tight_layout(); fig.savefig(out_dir / f'{path_id}-device-errors.png', dpi=140); plt.close(fig)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--pilot-dir', type=Path, default=Path('runs/native_pilot_resolved'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--instrumented-dir', type=Path)
    a = p.parse_args()
    result = analyze(a.trace, a.pilot_dir, a.out_dir, a.instrumented_dir)
    print(json.dumps({'matched_trace_rows': result['matched_trace_rows'],
                      'paths': len(result['path_summaries']), 'out_dir': str(a.out_dir)}, indent=2))


if __name__ == '__main__':
    main()
