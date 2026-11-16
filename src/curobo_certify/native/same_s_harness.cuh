// Diagnostic harness for native-vs-model falsification tests (curobo_certify).
// It is compiled AFTER the unmodified pinned upstream headers
// timeopt_common.cuh and timeopt_path_kernel.cuh (cuRobo c301cb00) with the same
// NVRTC options, and calls the upstream device functions progress_at_time,
// timing_state and sample_joint. The tails below mirror the expressions of
// sample_with_progress_kernel; bitwise agreement with production outputs is
// checked by scripts/native_same_s_falsify.py, never assumed.
#pragma once
namespace curobo_certify_harness {
using namespace curobo::trajectory::timeopt;

__device__ __forceinline__ void tail_values(const float *controls, const float *start,
                                            const float *goal, float s, const float *x,
                                            int dof, int knots, int start_offset,
                                            float *out, int stride, int idx) {
  const float speed = sqrtf(fmaxf(0.0f, x[0]));
  const int spans = knots + 3 - start_offset;
  const float coordinate = s * spans;
  const int segment = min(static_cast<int>(coordinate), spans - 1);
  for (int j = 0; j < dof; ++j) {
    float boundary[4], v[4];
    for (int k = 0; k < 4; ++k) boundary[k] = start[k * dof + j];
    sample_joint(controls, boundary, goal, 0, j, dof, knots, start_offset, segment,
                 coordinate - segment, v);
    const float values[4] = {v[0], v[1] * speed, v[2] * x[0] + 0.5f * v[1] * x[1],
                             speed * (v[3] * x[0] + 1.5f * v[2] * x[1] + 0.5f * v[1] * x[2])};
    for (int k = 0; k < 4; ++k) out[k * stride + idx * dof + j] = values[k];
  }
}

// Same requested physical time: production inversion + tail.
__global__ void progress_eval(const float *controls, const float *start,
                                         const float *goal, const float *timing,
                                         const float *polynomial, const float *nodes,
                                         const float *weights, const float *time,
                                         const float *requested, float *out, float *progress,
                                         int samples, int dof, int knots, int intervals,
                                         int start_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= samples) return;
  float x[3];
  const float s = progress_at_time(polynomial, timing, nodes, weights, time, 0, intervals,
                                   requested[idx], x);
  progress[idx] = s;
  tail_values(controls, start, goal, s, x, dof, knots, start_offset, out, samples * dof, idx);
}

// Same native progress s: timing_state at the caller's s + tail.
__global__ void same_s_eval(const float *controls, const float *start,
                                       const float *goal, const float *timing,
                                       const float *polynomial, const float *s_values,
                                       float *out, int samples, int dof, int knots,
                                       int start_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= samples) return;
  float x[3];
  const float s = s_values[idx];
  timing_state(polynomial, timing, 0, s, x);
  tail_values(controls, start, goal, s, x, dof, knots, start_offset, out, samples * dof, idx);
}
}  // namespace curobo_certify_harness
