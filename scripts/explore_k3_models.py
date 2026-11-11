#!/usr/bin/env python3
"""Per-residual exact Bernstein-tightness diagnostics for *serialized* q/x models.

No official campaign statistic is emitted. Time is not measured by this tool;
use the separately registered native/outward-CPU harness for latency.
"""
import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from curobo_certify.boundary_reference import Span, Boundaries, check_boundaries  # noqa: E402
from curobo_certify.exact_k3_reference import build_residuals  # noqa: E402
from curobo_certify.bernstein_tightness import bracket_polynomial_peak, diagnostic_dict  # noqa: E402


def analyze(blob: bytes, *, tolerance=Fraction(1, 1000000), max_depth=20, max_nodes=10000):
    model = json.loads(blob)
    if not isinstance(model, dict) or model.get('schema_version') != 1:
        raise ValueError('input must be model JSON object with schema_version=1')
    if model.get('basis') != 'local_power_ascending_u':
        raise ValueError('incorrect polynomial basis; no implicit conversion permitted')
    limits = model['limits']
    spans = [Span.from_values(x['left'], x['right'], x['q'], x['x'])
             for x in model['spans']]
    target = Boundaries.from_values(**model['targets'])
    boundary = check_boundaries(spans, target)
    records = []
    for index, span in enumerate(spans):
        for family, joint, power in build_residuals(
                span, limits['velocity'], limits['acceleration'], limits['jerk']):
            bracket = bracket_polynomial_peak(power, tolerance=tolerance,
                                              max_depth=max_depth, max_nodes=max_nodes)
            record = {'span_index': index, 'joint_index': joint, 'family': family,
                      's_left': str(span.left), 's_right': str(span.right)}
            record.update(diagnostic_dict(bracket))
            records.append(record)
    return {'scope': 'EXPLORATORY_EXACT_SERIALIZED_K3_MODEL_ONLY',
            'campaign_eligible': False, 'native_parity': 'NOT_EVALUATED',
            'runtime_latency': 'NOT_MEASURED', 'source_sha256': hashlib.sha256(blob).hexdigest(),
            'boundary_status': boundary.status, 'residual_records': records,
            'residual_count': len(records), 'span_count': len(spans)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path, help='explicit local-power K3 model JSON')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-depth', type=int, default=20)
    parser.add_argument('--max-nodes', type=int, default=10000)
    parser.add_argument('--tolerance', default='1/1000000', help='exact rational maximum gap')
    args = parser.parse_args()
    try:
        result = analyze(args.model.read_bytes(), tolerance=Fraction(args.tolerance),
                         max_depth=args.max_depth, max_nodes=args.max_nodes)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        result = {'scope': 'EXPLORATORY_EXACT_SERIALIZED_K3_MODEL_ONLY',
                  'status': 'INVALID_INPUT', 'reason': str(exc), 'campaign_eligible': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'{result.get("residual_count", 0)} residuals, status={result.get("status", "ANALYZED")}, file={args.out}')
    return 2 if result.get('status') == 'INVALID_INPUT' else 0


if __name__ == '__main__':
    raise SystemExit(main())
