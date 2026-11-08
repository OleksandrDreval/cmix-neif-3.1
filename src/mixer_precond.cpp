// Modified in November 2026 by Oleksandr Dreval (https://github.com/OleksandrDreval)
// for cmix-neif-3.1 (https://github.com/OleksandrDreval/cmix-neif-3.1).
// SPDX-License-Identifier: GPL-3.0-only
// The floating point of the layer-0 column preconditioning (mixer_precond.h) as an
// island translation unit with strict FP semantics (-ffp-model=precise
// -ffp-contract=off, no LTO/PGO), with the same recipe and for the same reason
// as adaptive_depth.cpp: one copy of the arithmetic, shared by encoder and
// decoder, with a fixed summation order.
#include "mixer_precond.h"
#include <immintrin.h>

namespace mixer_precond {

static inline float DetSqrt(float x) {
  return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
}

const float* L0ColScale(L0Col& c, const float* x, int n) {
  if (n > kMaxN0) n = kMaxN0;
  if (!c.init) {
    for (int i = 0; i < kMaxN0; ++i) { c.v[i] = 1.0f; c.r[i] = 1.0f; }
    c.init = true;
  }
  const float omb = 1.0f / (float)kL0ColN;
  const float beta = 1.0f - omb;
  for (int i = 0; i < n; ++i) c.v[i] = beta * c.v[i] + omb * (x[i] * x[i]);
  if ((c.n & 63) == 0) {
    float sum = 0.0f;   // fixed order: precise TU, no reassociation
    for (int i = 0; i < n; ++i) sum += c.v[i];
    const float ref = DetSqrt(sum / (float)n + 1.0e-12f);
    const float cap = (float)(32);
    const float lo = 1.0f / cap;
    int i = 0;
    const __m256 vref = _mm256_set1_ps(ref);
    const __m256 veps = _mm256_set1_ps(1.0e-12f);
    const __m256 vcap = _mm256_set1_ps(cap);
    const __m256 vlo  = _mm256_set1_ps(lo);
    for (; i + 8 <= n; i += 8) {
      __m256 vv = _mm256_add_ps(_mm256_loadu_ps(c.v + i), veps);
      __m256 vr = _mm256_div_ps(vref, _mm256_sqrt_ps(vv));
      vr = _mm256_min_ps(vr, vcap);
      vr = _mm256_max_ps(vr, vlo);
      _mm256_storeu_ps(c.r + i, vr);
    }
    for (; i < n; ++i) {
      float r = ref / DetSqrt(c.v[i] + 1.0e-12f);
      if (r > cap) r = cap;
      else if (r < lo) r = lo;
      c.r[i] = r;
    }
  }
  ++c.n;
  for (int i = 0; i < n; ++i) c.xt[i] = x[i] * c.r[i];
  return c.xt;
}

}  // namespace mixer_precond
