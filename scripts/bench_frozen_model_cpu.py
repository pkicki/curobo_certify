#!/usr/bin/env python3
"""Reproducible exploratory CPU timing of hooks on serialized, frozen q/x models.

NOT an end-to-end TimeOpt benchmark. No native GPU work, compilation,
CUDA sync, or query generation is timed. The default hook is exact Fraction
reference certification + proof replay and is not a production candidate.

Hook contract: function(data: bytes)->dict with status and optional proof_replay.
User implementations must include all required preprocessing and proof work.
"""
import argparse
import hashlib
import importlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def exact_reference_hook(blob: bytes) -> dict:
    from curobo_certify.boundary_reference import Boundaries, Span
    from curobo_certify.exact_k3_reference import certify_k3, replay_k3_leaves
    model = json.loads(blob)
    if model['schema_version'] != 1 or model['basis'] != 'local_power_ascending_u':
        raise ValueError('unsupported model format')
    spans = tuple(Span.from_values(s['left'], s['right'], s['q'], s['x'])
                  for s in model['spans'])
    targets = Boundaries.from_values(**model['targets'])
    limits = model['limits']
    arguments = (limits['velocity'], limits['acceleration'], limits['jerk'])
    result = certify_k3(spans, targets, *arguments)
    replay = (replay_k3_leaves(spans, targets, *arguments, result)
              if result.status == 'CERTIFIED_K3_REFERENCE' else None)
    if result.status == 'CERTIFIED_K3_REFERENCE' and not replay:
        raise RuntimeError('reference proof replay rejected certified output')
    return {'status': result.status, 'proof_replay': replay,
            'nodes': result.nodes_processed, 'scope': result.scope}


def get_hook(name):
    if name == 'exact_reference':
        return exact_reference_hook
    if ':' not in name:
        raise ValueError('hook must be exact_reference or module:function')
    module, function = name.rsplit(':', 1)
    result = getattr(importlib.import_module(module), function)
    if not callable(result):
        raise ValueError('hook is not callable')
    return result


def benchmark_model(blob: bytes, hooks, warmups: int, repeats: int, seed: int):
    if warmups < 0 or repeats < 1:
        raise ValueError('invalid benchmark repeat counts')
    if not hooks:
        raise ValueError('at least one hook required')
    digest = hashlib.sha256(blob).hexdigest()
    results = []
    for label, fn in hooks.items():
        for _ in range(warmups):
            if not isinstance(fn(blob), dict):
                raise TypeError('hook must return a status dict')
    rng = random.Random(seed)
    for rep in range(repeats):
        labels = list(hooks)
        if rep % 2:
            labels.reverse()
        if len(labels) > 2:
            rng.shuffle(labels)
        for label in labels:
            # A distinct bytes object per invocation. Hook parses/clones as
            # needed; timing starts before hook and ends after a CPU result.
            input_copy = bytes(bytearray(blob))
            start = time.perf_counter_ns()
            output = hooks[label](input_copy)
            elapsed = time.perf_counter_ns() - start
            if not isinstance(output, dict) or not isinstance(output.get('status'), str):
                raise TypeError('hook must return dict with string status')
            if hashlib.sha256(input_copy).hexdigest() != digest:
                raise RuntimeError('hook mutated frozen input bytes')
            results.append({'repeat': rep, 'hook': label, 'elapsed_ns': elapsed,
                            'status': output['status'],
                            'proof_replay': output.get('proof_replay'),
                            'scope': output.get('scope', 'UNDECLARED')})
    medians = {label: statistics.median([x['elapsed_ns'] for x in results if x['hook'] == label])
               for label in hooks}
    return {'source_sha256': digest, 'benchmark_scope': 'EXPLORATORY_FROZEN_MODEL_CPU_ONLY',
            'preprocessing_contract': 'hook parses/constructs its input inside timer',
            'native_linked': False, 'campaign_eligible': False,
            'warmups_per_hook': warmups, 'repetitions_per_hook': repeats,
            'hook_median_ns': medians, 'raw_records': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', type=Path)
    parser.add_argument('--hook', action='append', default=[],
                        help='label=module:function; default exact_reference')
    parser.add_argument('--warmups', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    named = args.hook or ['exact_reference=exact_reference']
    hooks = {}
    for spec in named:
        if '=' not in spec:
            parser.error('expected --hook label=module:function')
        label, entry = spec.split('=', 1)
        if not label or label in hooks:
            parser.error('hook labels must be unique and nonempty')
        hooks[label] = get_hook(entry)
    results = [dict(source_path=str(path), **benchmark_model(path.read_bytes(), hooks,
                                                               args.warmups, args.repeats,
                                                               args.seed + i))
               for i, path in enumerate(args.models)]
    output = {'schema_version': 1, 'campaign_grade': False,
              'latency_scope': 'CPU_EXPLORATORY_FROZEN_SERIALIZED_MODEL',
              'excludes': ['native planner', 'TimeOpt optimization', 'native adapter parity',
                           'GPU work', 'GPU transfers', 'end-to-end request'],
              'results': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(f'Wrote exploratory CPU timing for {len(results)} model file(s) to {args.out}')


if __name__ == '__main__':
    main()
