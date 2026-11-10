#!/usr/bin/env python3
"""Create a compact offline-report payload from retained pilot arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--proofs', type=Path)
    parser.add_argument('--event-audit', type=Path)
    args = parser.parse_args()
    proof_path = args.proofs or (args.pilot_dir / 'k3_model_diagnostics.json')
    proof_by_id = {}
    if proof_path.exists():
        proof_doc = json.loads(proof_path.read_text())
        proof_by_id = {x['path_id']: x for x in proof_doc.get('paths', [])}
    event_by_id = {}
    if args.event_audit and args.event_audit.exists():
        event_doc = json.loads(args.event_audit.read_text())
        if event_doc.get('source_id'):
            event_by_id[event_doc['source_id']] = event_doc
        else:
            for record_path in event_doc.get('records', []):
                candidate = Path(record_path)
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                if candidate.exists():
                    event_record = json.loads(candidate.read_text())
                    event_by_id[event_record['source_id']] = event_record
    paths = []
    for meta_path in sorted(args.pilot_dir.glob('pilot-*.json')):
        meta = json.loads(meta_path.read_text())
        row = {k: meta.get(k) for k in ('source_id', 'status', 'reason', 'evidence_kind', 'native_duration_seconds',
              'source_query_sha256', 'geometry_hash', 'timing_hash', 'robot_config_sha256', 'scene_config_sha256')}
        row['path_id'] = meta_path.stem
        row['parity'] = meta.get('native_adapter_parity', {}).get('status', 'NOT_AVAILABLE')
        proof_row = proof_by_id.get(meta_path.stem)
        if proof_row:
            row['model_k3_by_depth'] = [
                {'depth': b['max_depth'], 'status': b['k3_status'],
                 'nodes': b['processed_nodes'], 'leaves': b['proof_leaves'],
                 'replay': b['replay_pass'], 'native_linkage': proof_row.get('native_linkage')}
                for b in proof_row.get('budgets', [])
            ]
        if meta.get('planner_success'):
            arr_path = args.pilot_dir / Path(meta['arrays_path']).name
            raw = arr_path.read_bytes()
            row['arrays_sha256'] = hashlib.sha256(raw).hexdigest()
            if row['arrays_sha256'] != meta['arrays_sha256']:
                raise SystemExit(f'array hash mismatch: {arr_path}')
            with np.load(arr_path, allow_pickle=False) as a:
                times = a['diagnostic_8193_times']
                states = a['diagnostic_8193_states']
                stride = 8
                ids = list(range(0, len(times), stride))
                if ids[-1] != len(times) - 1:
                    ids.append(len(times) - 1)
                row['samples'] = {
                    'time_s': times[ids].astype(float).tolist(),
                    'position': states[0, ids].astype(float).tolist(),
                    'velocity': states[1, ids].astype(float).tolist(),
                    'acceleration': states[2, ids].astype(float).tolist(),
                    'jerk': states[3, ids].astype(float).tolist(),
                }
                row['limits'] = {
                    family: {'lower': a[f'{family}_limits'][0].astype(float).tolist(),
                             'upper': a[f'{family}_limits'][1].astype(float).tolist()}
                    for family in ('velocity', 'acceleration', 'jerk')
                }
                row['endpoint_positions'] = [a['requested_start'][0].astype(float).tolist(),
                                             a['requested_goal'][0].astype(float).tolist()]
                dense_meta = meta.get('native_adapter_parity', {}).get('dense_diagnostics', [{}, {}])
                event_audit = event_by_id.get(meta.get('source_id'))
                if event_audit:
                    row['event_audit_status'] = event_audit.get('native_adapter_parity', {}).get('dense_diagnostics', [{}, {}])[1].get('observed_event_count')
                    row['event_audit_expected'] = event_audit.get('native_adapter_parity', {}).get('dense_diagnostics', [{}, {}])[1].get('expected_event_count')
                    dense_meta = event_audit['native_adapter_parity']['dense_diagnostics']
                path_knots = {i / (a['effective_geometry'].shape[0] - 3)
                              for i in range(1, a['effective_geometry'].shape[0] - 3)}
                timing_knots = set(a['timing_knot_vector'].astype(float).tolist())
                row['knot_events'] = [
                    {'family': e.get('family') or (
                         'both' if any(abs(e['knot'] - x) < 1e-12 for x in path_knots)
                         and any(abs(e['knot'] - x) < 1e-12 for x in timing_knots)
                         else 'path' if any(abs(e['knot'] - x) < 1e-12 for x in path_knots)
                         else 'timing'), 'progress': e['knot'],
                     'side': e['side'], 'time_s': e['time'],
                     'sampled_progress': e.get('sampled_progress'),
                     'side_verified': e.get('side_verified')}
                    for e in dense_meta[1].get('events', [])
                ]
                row['sample_count'] = len(times)
                row['joint_count'] = int(states.shape[-1])
                row['duration_s'] = float(a['native_duration'][0])
                row['arrays_file'] = arr_path.name
        paths.append(row)
    out = {'evidence_kind': 'real_native_planner_timeopt_pilot', 'campaign_measurements': False,
           'upstream_sha': 'c301cb00a87012c2808b1a60649c4065c9b3c95a',
           'hardware': 'NVIDIA GeForce RTX 3090', 'pilot_paths': paths,
           'latency': 'NOT MEASURED',
           'warning': 'Sampled plots are diagnostics, not certificates. All eight Panda adapters fail strict parity; serialized-model proof results are not linked native certificates. UR5e has no verified model config in the pin.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {len(paths)} pilot rows to {args.out}')


if __name__ == '__main__':
    main()
