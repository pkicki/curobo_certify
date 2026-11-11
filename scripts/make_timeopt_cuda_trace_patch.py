#!/usr/bin/env python3
"""Produce an opt-in pinned-cuRobo CUDA printf patch without editing upstream.

Apply ONLY to a disposable, separately built checkout; never to the pinned
production dependency. This diagnostic build is not suitable for timing.
"""
import argparse
import difflib
import hashlib
import subprocess
from pathlib import Path

PIN = 'c301cb00a87012c2808b1a60649c4065c9b3c95a'
REL_PATH = ('curobo/_src/curobolib/kernels/trajectory/timeopt/'
            'timeopt_path_kernel.cuh')
BLOB = '714d99c58d2e0f083a8c9fafdc68bef0b0a63f8f'
LOOP_ANCHOR = '''    for (int k = 0; k < 4; ++k) {
      sample_joints[k * plane + idx * dof + j] ='''


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def patch_source(source: str, sample_index: int, joint: int) -> str:
    if 'TIMEOPT_TRACE idx=' in source:
        raise ValueError('CUDA source already instrumented; refusing duplicate patch')
    if sample_index < 0 or joint < 0:
        raise ValueError('index and joint must be nonnegative')
    if source.count(LOOP_ANCHOR) != 1 or source.count('#include "timeopt/timeopt_path_helpers.cuh"') != 1:
        raise ValueError('pinned upstream CUDA anchors missing or ambiguous')
    # Print arguments as exact IEEE-754 binary32 bit patterns. This exposes
    # device-side operands from the actual evaluation site. Recomputation of
    # subterms is labelled diagnostic and may have a different contraction.
    payload = f'''    // Diagnostic-only CUDA print. DO NOT use this build for latency.
    // The operands and output are from the native sample kernel; diagnostic
    // products below are recomputed and do not prove original FMA grouping.
    if (idx == {sample_index} && j == {joint}) {{
      printf("TIMEOPT_TRACE idx=%d j=%d t=%08x s=%08x coord=%08x local=%08x seg=%d "
             "x0=%08x x1=%08x x2=%08x q0=%08x q1=%08x q2=%08x q3=%08x "
             "speed=%08x native_jerk=%08x\\n",
             idx, j, __float_as_uint(requested), __float_as_uint(s),
             __float_as_uint(coordinate), __float_as_uint(coordinate - segment), segment,
             __float_as_uint(x[0]), __float_as_uint(x[1]), __float_as_uint(x[2]),
             __float_as_uint(v[0]), __float_as_uint(v[1]),
             __float_as_uint(v[2]), __float_as_uint(v[3]),
             __float_as_uint(speed), __float_as_uint(values[3]));
      const float t1_diag = v[3] * x[0];
      const float t2_diag = 1.5f * v[2] * x[1];
      const float t3_diag = 0.5f * v[1] * x[2];
      printf("TIMEOPT_TERMS idx=%d j=%d t1=%08x t2=%08x t3=%08x\\n",
             idx, j, __float_as_uint(t1_diag), __float_as_uint(t2_diag),
             __float_as_uint(t3_diag));
    }}
'''
    return source.replace('#include "timeopt/timeopt_path_helpers.cuh"',
                          '#include "timeopt/timeopt_path_helpers.cuh"\n#include <cstdio>', 1).replace(
                          LOOP_ANCHOR, payload + LOOP_ANCHOR, 1)


def create_patch(checkout: Path, sample_index: int, joint: int) -> str:
    checkout = checkout.resolve()
    head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if head != PIN:
        raise ValueError(f'upstream SHA mismatch {head} != {PIN}')
    file = checkout / REL_PATH
    old_bytes = file.read_bytes()
    if git_blob_sha(old_bytes) != BLOB:
        raise ValueError(f'original native kernel blob differs from pinned {BLOB}')
    # Refuse files with uncommitted edits, even if they happen to match content.
    modified = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--', REL_PATH], text=True)
    if modified.strip():
        raise ValueError('upstream kernel is modified; refusing to operate')
    old = old_bytes.decode('utf-8')
    new = patch_source(old, sample_index, joint)
    return ''.join(difflib.unified_diff(old.splitlines(keepends=True),
                                       new.splitlines(keepends=True),
                                       fromfile='a/' + REL_PATH, tofile='b/' + REL_PATH))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream-checkout', type=Path, required=True)
    parser.add_argument('--sample-index', type=int, required=True,
                        help='flattened idx = batch_idx * n_samples + sample_idx')
    parser.add_argument('--joint', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True, help='out-of-tree patch file')
    args = parser.parse_args()
    patch = create_patch(args.upstream_checkout, args.sample_index, args.joint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(patch)
    print(f'Wrote diagnostic patch {args.out}; source checkout unchanged')


if __name__ == '__main__':
    main()
