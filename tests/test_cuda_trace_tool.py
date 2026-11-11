import hashlib
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_timeopt_cuda_trace_patch import patch_source, git_blob_sha
from parse_timeopt_cuda_trace import decode_lines


MINIMAL = '''#include "timeopt/timeopt_path_helpers.cuh"
void sample() {
const float values[4] = {0,0,0,0};
    for (int k = 0; k < 4; ++k) {
      sample_joints[k * plane + idx * dof + j] = 0;
    }
}
'''


class TraceTests(unittest.TestCase):
    def test_patch_only_one_print_site(self):
        mod = patch_source(MINIMAL, 7, 2)
        self.assertIn('((idx == 7) && (j == 2))', mod)
        self.assertIn('native_jerk=%08x', mod)
        self.assertNotIn('#include <', mod)
        self.assertEqual(mod.count('TIMEOPT_TRACE idx='), 1)
        self.assertIn('sample_joints[k * plane + idx * dof + j]', mod)
        with self.assertRaises(ValueError):
            patch_source(mod, 7, 2)

    def test_patch_multiple_indices(self):
        mod = patch_source(MINIMAL, [2, 9], 0)
        self.assertIn('(idx == 2 || idx == 9) && (j == 0)', mod)

    def test_all_joint_trace_mode(self):
        mod = patch_source(MINIMAL, [3], -1)
        self.assertIn('((idx == 3) && (true))', mod)

    def test_fixed_progress_override_is_explicit_and_bounded(self):
        from make_timeopt_cuda_trace_patch import patch_source
        source = MINIMAL.replace('void sample() {', '''void sample() {
  float x[3];
  const float *polynomial, *output_timing, *nodes, *time_weights, *time;
  int b, intervals;
  float requested;
  const float s = progress_at_time(polynomial, output_timing, nodes, time_weights,
                                 time, b, intervals, requested, x);''')
        patched = patch_source(source, 7, 2, 0.379210323)
        self.assertIn('s = 0.379210323f;', patched)
        self.assertIn('timing_state(polynomial, output_timing, b, s, x);', patched)
        with self.assertRaises(ValueError):
            patch_source(source, 7, 2, 1.1)

    def test_invalid_anchor_fails_closed(self):
        with self.assertRaises(ValueError):
            patch_source('broken source', 0, 0)
        with self.assertRaises(ValueError):
            patch_source(MINIMAL, -1, 1)

    def test_git_blob_hash(self):
        data = b'hello\n'
        self.assertEqual(git_blob_sha(data), hashlib.sha1(b'blob 6\x00hello\n').hexdigest())

    def test_parse_bits_and_terms(self):
        trace = ('[0] TIMEOPT_TRACE idx=5 j=1 t=3f800000 s=3e800000 coord=3f000000 '
                 'local=3e000000 seg=10 x0=3f800000 x1=00000000 x2=00000000 '
                 'q0=3f000000 q1=3f800000 q2=00000000 q3=00000000 '
                 'speed=3f800000 native_jerk=c1200000\n'
                 'TIMEOPT_TERMS idx=5 j=1 t1=3f800000 t2=00000000 t3=80000000\n')
        result = decode_lines(trace)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['seg'], 10)
        self.assertEqual(result[0]['native_jerk'], -10.0)
        self.assertEqual(result[0]['t3_bits'], '80000000')
        self.assertEqual(struct.pack('!f', result[0]['t3']), bytes.fromhex('80000000'))
        self.assertTrue(result[0]['terms_present'])
        self.assertTrue(result[0]['trace_present'])

    def test_unrelated_logs_ignored(self):
        self.assertEqual(decode_lines('Hello there\n...\n'), [])

    def test_repeated_sample_index_preserves_call_occurrences(self):
        one = ('TIMEOPT_TRACE idx=5 j=1 t=3f800000 s=3e800000 coord=3f000000 '
               'local=3e000000 seg=10 x0=3f800000 x1=00000000 x2=00000000 '
               'q0=3f000000 q1=3f800000 q2=00000000 q3=00000000 '
               'speed=3f800000 native_jerk=c1200000\n')
        records = decode_lines(one + one)
        self.assertEqual(len(records), 2)


if __name__ == '__main__':
    unittest.main()
