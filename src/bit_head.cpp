// Modified in November 2026 by Oleksandr Dreval (https://github.com/OleksandrDreval)
// for cmix-neif-3.1 (https://github.com/OleksandrDreval/cmix-neif-3.1).
// SPDX-License-Identifier: GPL-3.0-only
// Per-bit LSTM correction head.  The contract is described in the header of
// the same name.
//
// This translation unit MUST be compiled with value-safe floating point
// (-ffp-model=precise -ffp-contract=off, see the makefile rule `bitl`) and
// its entry points are noinline: encode and decode then execute the identical
// machine code, so the head stays in lockstep by construction.
//
// ATTRIBUTION.  The AVX2 kernels Exp8/Sigmoid8/Tanh8 and the fixed-order
// DotPadded reduction below are ported line by line from
// src/models/bitlstm32-head.cpp of cmix-obias (David Freelan, GPL-3.0);
// Exp8 is an altered version of exp256_ps of avx_mathfun.h (Giovanni
// Garberoglio, 2012), itself based on sse_mathfun.h (Julien Pommier, 2007);
// zlib licence, notice in third_party/mathfun-LICENSE.txt of the source
// package; its polynomial is the Cephes expf of Stephen L. Moshier.  The forward
// architecture (in_proj -> SiLU -> LSTMCell -> linear, logit-additive
// correction after Discretize), several input features and the offline
// training on a trace follow the same model.  The online backward pass, the
// optimiser, the state projections and the parameters are our own.

#include "meld.h"
#include "bit_head.h"
#include "cold_size.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "isa_floor.h"
#include <immintrin.h>

#include "bit_head_weights.h"

namespace bit_head {
namespace {

// Logit-space normaliser.  Same constant as the reference head: it only has
// to bring the stage-1 row, the terminal logit and the SSE correction into
// the same O(1) range so a single learning rate fits all of them.
constexpr float kTScale = 12.203f;

// Feature block that is always present, on top of the stage-1 row:
//   +0        m1raw / T
//   +1..+3    t/T, (t/T)^2, sgn(t)
//   +4..+11   bit position one-hot(8)
constexpr int kBaseExtra = 12;
// Optional SSE block (feature bit 0, off in the release configuration): sd/T and
// mean|sd| over 256 / T.
constexpr int kSseExtra = 2;

constexpr int kNTf = 0;

inline float Sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float Sgn(float x) { return (x > 0.0f) ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

// ---------------------------------------------------------------------------
// AVX2 kernels (ported, see ATTRIBUTION above)
// ---------------------------------------------------------------------------
// 8-lane expf: altered exp256_ps of avx_mathfun.h (see the attribution
// above), Cephes polynomial.  Fixed coefficients and fixed operation order,
// so it is deterministic.
inline __m256 Exp8(__m256 x) {
  const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
  const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
  const __m256 c1 = _mm256_set1_ps(0.693359375f);
  const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
  const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
  const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
  const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
  const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
  const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);
  const __m256 one = _mm256_set1_ps(1.0f);
  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);
  __m256 fx = _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5f));
  fx = _mm256_floor_ps(fx);
  x = _mm256_fnmadd_ps(fx, c1, x);
  x = _mm256_fnmadd_ps(fx, c2, x);
  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = p0;
  y = _mm256_fmadd_ps(y, x, p1);
  y = _mm256_fmadd_ps(y, x, p2);
  y = _mm256_fmadd_ps(y, x, p3);
  y = _mm256_fmadd_ps(y, x, p4);
  y = _mm256_fmadd_ps(y, x, p5);
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, one);
  const __m256i imm = _mm256_add_epi32(_mm256_cvttps_epi32(fx),
                                       _mm256_set1_epi32(0x7f));
  const __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(imm, 23));
  return _mm256_mul_ps(y, pow2n);
}
inline __m256 Sigmoid8(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  return _mm256_div_ps(one, _mm256_add_ps(one,
      Exp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}
inline __m256 Tanh8(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 e = Exp8(_mm256_mul_ps(x, _mm256_set1_ps(-2.0f)));
  return _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));
}

// Deterministic dense dot product with a fixed-order horizontal reduction.
inline float Hsum8(__m256 acc) {
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}
inline float DotPadded(const float* w, const float* x, int n_padded) {
  __m256 acc = _mm256_setzero_ps();
  for (int j = 0; j < n_padded; j += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + j), _mm256_loadu_ps(x + j), acc);
  }
  return Hsum8(acc);
}
inline float Dot16Preloaded(const float* w, __m256 x0, __m256 x1) {
  __m256 acc = _mm256_setzero_ps();
  acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + 0), x0, acc);
  acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + 8), x1, acc);
  return Hsum8(acc);
}
inline float Dot32Preloaded(const float* w, __m256 x0, __m256 x1, __m256 x2, __m256 x3) {
  __m256 acc = _mm256_setzero_ps();
  acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + 0), x0, acc);
  acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + 8), x1, acc);
  acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + 16), x2, acc);
  acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + 24), x3, acc);
  return Hsum8(acc);
}

// Deterministic splitmix64 (Steele, Lea and Flood; public-domain reference
// code by Sebastiano Vigna): the initial weights must be identical in encoder
// and decoder, so no library RNG and no time/address dependence.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t Next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  // Uniform in [-b, b], 24 mantissa bits.
  float Uniform(float b) {
    const uint32_t r = (uint32_t)(Next() >> 40);  // 24 bits
    const float u = (float)r * (1.0f / 16777216.0f);  // [0,1)
    return (2.0f * u - 1.0f) * b;
  }
};

struct Head {
  static constexpr int cells = 32;
  // stage-1 row width: the row plus the 13 columns of the fast-weight logit,
  // the bit position and the path state, and the semantic-cache columns
  static constexpr int n1 = 22 + 13 + 4;
  // the bit-position column of the row (second column of MlAppend)
  static constexpr int kBposCol = 22 + 1;
  static constexpr int nsse = ((0) & kFeatSse) ? kSseExtra : 0;
  // every column but column 0 and kBposCol
  static constexpr int n1keep = n1 - 2;
  static constexpr int nhid = 32;
  static constexpr int ntf = kNTf;
  static constexpr int nzone = 0;   // gate-zone one-hot: not used
  // Layout of the feature row:
  //   [0 .. n1keep-1]              stage-1 row (selected columns)
  //   [n1keep+0]                   m1raw
  //   [n1keep+1 .. n1keep+3]       t/T, (t/T)^2, sgn(t)
  //   [n1keep+4 .. n1keep+11]      bit-position one-hot(8)
  //   [n1keep+12, n1keep+13]       SSE pair (only with feat & kFeatSse)
  //   [kPrev]                      prev_bit
  //   [kHid ..] [kTf ..] [kZone ..]   optional blocks, empty here
  static constexpr int kPrev = n1keep + kBaseExtra + nsse;
  static constexpr int kHid = kPrev + 1;
  static constexpr int nin =                // features without prev_bit
      n1keep + kBaseExtra + nsse + nhid + ntf + nzone;
  static constexpr int inw = ((nin + 1) + 7) & ~7;   // padded to 8

  static constexpr float clamp = (float)(8000 / 1000.0);
  static constexpr int seq = 1024;
  static constexpr int xscale = (0) ? 1 : 0;

  // ---- weights ----
  float* win = nullptr;    // cells x inw
  float* bin = nullptr;    // cells
  float* wih = nullptr;    // 4*cells x cells
  float* whh = nullptr;    // 4*cells x cells
  float* bg = nullptr;     // 4*cells
  float* wout = nullptr;   // cells
  float bout = 0.0f;

  // ---- recurrent state ----
  float* h = nullptr;
  float* c = nullptr;
  float prev_bit = 0.0f;      // 0.5 from Init()
  unsigned long long i = 0;   // coded-bit index

  // ---- per-bit scratch, live between Adjust() and Observe() ----
  float* x = nullptr;      // inw, feature row incl. prev_bit
  float* z = nullptr;      // cells, in_proj pre-activation
  float* a = nullptr;      // cells, SiLU(z)
  float* pre = nullptr;    // 4*cells
  float delta = 0.0f;
  float t = 0.0f;
  int skip = 0;            // override bit: forward only, no learning

  // ---- per-byte channel ----
  // Registered by the coder before Adjust(); constant across the eight bits of
  // a byte.  Pure observation - nothing the head computes from them re-enters
  // the predictor.
  const float* aux_hid = nullptr;   // TF residual stream
  int aux_n_hid = 0;
  const float* aux_tf = nullptr;    // kTfAll per-byte scalars
  int aux_n_tf = 0;
  const unsigned int* aux_zone = nullptr;   // gate zone of this byte (not read in the release configuration)
  int hid_proj = 0;                 // 1 = fixed +-1/sqrt(n) sketch
  float* hidP = nullptr;            // nhid x aux_n_hid
  float* hidC = nullptr;            // nhid, the sketch OF THIS BYTE
  unsigned long long hid_byte = 0;   // byte hidC belongs to, ~0 from Init()
  int slot = 0;                     // weight slot in force (0 metablock, 1 text 1, 2 text 2)
  int seen3 = 0;                    // slot 0: a HDRBLOCK bit was coded
  unsigned long long trig = 0;      // 1 + head bit index where the next slot's zone began, 0 = not yet
  static constexpr bool SlotHas(int k) { return kBlobTafelN[k] != 0; }
  void NextSlot();

  // window length in coded bits
  static constexpr int wlen = (seq > 0) ? seq : 64;
  int nbuf = 0;      // steps buffered so far
  int nparams = 0;

  void Build();
  void LoadBlob();
  void Forward();
  void Free();
};

Head g_head;
bool g_init_done = false;
// The stamp tool checks that all slot blobs share one geometry and writes it
// here, so a blob of another form stops the build, not the run at the slot
// switch (error E51 would only come with slot 2, more than half way through).
static_assert(7 == 0 ||
                  (32 == Head::cells && 81 == Head::nin &&
                   88 == Head::inw && 39 == Head::n1),
              "the stamped slot blobs have another geometry (cells/inputs/input width/stage-1 width) than this head");

COLD_SIZE_ATTR float* Alloc(int n) {
  float* p = (float*)std::calloc((size_t)n > 0 ? (size_t)n : 1, sizeof(float));
  return p;
}

COLD_SIZE_ATTR void Head::Free() {
  std::free(win); std::free(bin); std::free(wih); std::free(whh);
  std::free(bg); std::free(wout);
  std::free(h); std::free(c);
  std::free(x); std::free(z); std::free(a);
  std::free(pre);
  std::free(hidP); std::free(hidC);
  std::memset(this, 0, sizeof(*this));
}

COLD_SIZE_ATTR void Head::Build() {
  const int C = cells;

  // Build guard for the state-vector block.
  {
    if (nhid > aux_n_hid) {
      msg_stop(44);
      std::fflush(stderr);
      std::abort();
    }
  }
  hid_proj = (nhid < aux_n_hid) ? 1 : 0;
  if (hid_proj) {
    hidP = Alloc(nhid * aux_n_hid);
    hidC = Alloc(nhid);
    Rng prng(0x4849445052ULL);   // "HIDPR"
    const float sc = 1.0f / std::sqrt((float)aux_n_hid);
    for (int r = 0; r < nhid; ++r) {
      for (int cc = 0; cc < aux_n_hid; ++cc) {
        hidP[cc * nhid + r] = (prng.Uniform(1.0f) >= 0.0f) ? sc : -sc;
      }
    }
  }

  win = Alloc(C * inw);   bin = Alloc(C);
  wih = Alloc(4 * C * C); whh = Alloc(4 * C * C);
  bg = Alloc(4 * C);      wout = Alloc(C);
  h = Alloc(C); c = Alloc(C);
  x = Alloc(inw); z = Alloc(C); a = Alloc(C);
  pre = Alloc(4 * C);

  nparams = C * (nin + 1) + C + 8 * C * C + 4 * C + C + 1;
  if (SlotHas(0)) LoadBlob();
}

COLD_SIZE_ATTR __attribute__((noinline)) void Head::LoadBlob() {
  const unsigned char* ps = kBlobW + kSlotOff[slot];
  const unsigned char* ex = kBlobW + kBlobN + kSlotExcOff[slot];
  const unsigned H = kSlotH[slot];
  const unsigned char* cm = kSlotCols[slot];
  float* const d[7] = {win, bin, wih, whh, bg, wout, &bout};
  static constexpr unsigned n[7] = {(unsigned)(cells * inw), (unsigned)cells, (unsigned)(4 * cells * cells),
                                    (unsigned)(4 * cells * cells), (unsigned)(4 * cells), (unsigned)cells, 1u};
  for (int k = 0; k < 7; ++k)
    for (unsigned i = 0; i < n[k]; ++i) {
      const unsigned q = i / (unsigned)cells, m = i % (unsigned)cells;
      float* p;
      bool live;
      if (k == 0) {   // win column by column
        p = win + m * (unsigned)inw + q;
        live = m < H && (cm[q >> 3] >> (q & 7) & 1);
      } else {
        p = d[k] + i;
        live = q % (unsigned)cells < H && m < H;
      }
      std::uint32_t u = 0;
      if (live) {
        const unsigned b = *ps++;
        const unsigned ec = b >> 4;
        const unsigned ev = (ec < 15u) ? (ec + 116u) : *ex++;
        u = (std::uint32_t)(b & 0x08) << 28 | (std::uint32_t)ev << 23 | (std::uint32_t)(b & 0x07) << 20;
      }
      __builtin_memcpy(p, &u, sizeof(u));
    }
}

// The switch to the next weight slot, called by Adjust() on the first bit of
// a window (h and c are reset right behind it, see the header).
COLD_SIZE_ATTR __attribute__((noinline)) void Head::NextSlot() {
  ++slot;
  trig = 0;
  prev_bit = 0.5f;
  if (SlotHas(slot)) {
    LoadBlob();
  }
}

void Head::Forward() {
  const int C = cells;
  const int H = kSlotH[slot];
  for (int r = 0; r < H; ++r) z[r] = DotPadded(win + r * inw, x, inw) + bin[r];
  for (int r = 0; r < H; r += 8) {
    const __m256 zv = _mm256_loadu_ps(z + r);
    const __m256 s = Sigmoid8(zv);
    _mm256_storeu_ps(a + r, _mm256_mul_ps(zv, s));
  }
  const __m256 a0 = _mm256_loadu_ps(a + 0);
  const __m256 a1 = _mm256_loadu_ps(a + 8);
  const __m256 hp0 = _mm256_loadu_ps(h + 0);
  const __m256 hp1 = _mm256_loadu_ps(h + 8);
  if (H == 16) {
    for (int g = 0; g < 4; ++g) {
      for (int r = 0; r < 16; ++r) {
        const int k = g * C + r;
        pre[k] = Dot16Preloaded(wih + k * C, a0, a1) +
                 Dot16Preloaded(whh + k * C, hp0, hp1) +
                 bg[k];
      }
    }
  } else {
    const __m256 a2 = _mm256_loadu_ps(a + 16);
    const __m256 a3 = _mm256_loadu_ps(a + 24);
    const __m256 hp2 = _mm256_loadu_ps(h + 16);
    const __m256 hp3 = _mm256_loadu_ps(h + 24);
    for (int k = 0; k < 4 * C; ++k) {
      pre[k] = Dot32Preloaded(wih + k * C, a0, a1, a2, a3) +
               Dot32Preloaded(whh + k * C, hp0, hp1, hp2, hp3) +
               bg[k];
    }
  }
  for (int r = 0; r < H; r += 8) {
    const __m256 vi = Sigmoid8(_mm256_loadu_ps(pre + 0 * C + r));
    const __m256 vf = Sigmoid8(_mm256_loadu_ps(pre + 1 * C + r));
    const __m256 vg = Tanh8(_mm256_loadu_ps(pre + 2 * C + r));
    const __m256 vo = Sigmoid8(_mm256_loadu_ps(pre + 3 * C + r));
    const __m256 cn = _mm256_fmadd_ps(vf, _mm256_loadu_ps(c + r),
                                      _mm256_mul_ps(vi, vg));
    const __m256 th = Tanh8(cn);
    _mm256_storeu_ps(c + r, cn);
    _mm256_storeu_ps(h + r, _mm256_mul_ps(vo, th));
  }
  delta = DotPadded(wout, h, H) + bout;
}

}  // namespace

COLD_SIZE_ATTR bool Init() {
  if (g_init_done) return true;
  g_init_done = true;
  Head& s = g_head;
  s.prev_bit = 0.5f;
  s.hid_byte = ~0ULL;
  s.Build();
  return true;
}

int Cells() { return g_head.cells; }
int NumParams() { return g_head.nparams; }

void SetAux(const float* hid, int n_hid) {
  g_head.aux_hid = hid;
  g_head.aux_n_hid = (hid != nullptr) ? n_hid : 0;
}
void SetTf(const float* tf, int n_tf) {
  g_head.aux_tf = tf;
  g_head.aux_n_tf = (tf != nullptr) ? n_tf : 0;
}
void SetZone(const unsigned int* zone) { g_head.aux_zone = zone; }

void Shutdown() {
  if (g_init_done) g_head.Free();
  g_init_done = false;
}

__attribute__((noinline))
unsigned int Adjust(unsigned int base_p, const float* stage1, int n_stage1,
                    float m1raw, int override_active) {
  Head& s = g_head;
  if (__builtin_expect(s.i == 0, 0) && n_stage1 != Head::n1) {
    msg_stop(53);
    std::fflush(stderr);
    std::abort();
  }
  if (s.slot == 0) {
    if (tf_gate::ctx.zone == 3u) s.seen3 = 1;
    else if (s.seen3 && s.trig == 0) s.trig = s.i + 1;
  } else if (s.slot == 1 && s.trig == 0 &&
             tf_gate::ctx.pages >= (unsigned int)143227) {
    s.trig = s.i + 1;
  }
  if (s.trig != 0 && s.nbuf == 0) s.NextSlot();

  // Sequence reset: the recurrent state is zeroed at the start of every window of seq bits.
  if (s.nbuf == 0) {
    for (int r = 0; r < s.cells; ++r) { s.h[r] = 0.0f; s.c[r] = 0.0f; }
  }

  // ---- features ----
  const float p1 = (float)base_p / 65536.0f;
  const float t = std::log(p1 / (1.0f - p1));
  s.t = t;
  const float inv_t = 1.0f / kTScale;
  float* x = s.x;
  for (int j = 0; j < s.inw; ++j) x[j] = 0.0f;
  const int n1 = Head::n1keep;
  const float xs = s.xscale ? inv_t : 1.0f;
  // every column but column 0 and the bit-position column,
  // compacted in column order.
  {
    int w = 0;
    for (int j = 1; j < Head::n1; ++j) {
      if (j == Head::kBposCol) continue;
      const float v = (j < n_stage1) ? stage1[j] : 0.0f;
      x[w++] = v * xs;
    }
  }
  x[n1 + 0] = m1raw * xs;
  const float tl = t * inv_t;
  x[n1 + 1] = tl;
  x[n1 + 2] = tl * tl;
  x[n1 + 3] = Sgn(t);
  {
    // the true bit position (bits of the byte already coded)
    int bp = (int)stage1[Head::kBposCol];
    if (bp < 0) bp = 0;
    if (bp > 7) bp = 7;
    x[n1 + 4 + bp] = 1.0f;
  }
  x[Head::kPrev] = s.prev_bit;
  if (s.aux_hid != nullptr) {
    if (s.hid_proj) {
      const unsigned long long b = s.i >> 3;
      if (b != s.hid_byte) {
        s.hid_byte = b;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (int cc = 0; cc < s.aux_n_hid; ++cc) {
          const float* pr = s.hidP + (size_t)cc * Head::nhid;
          const __m256 vh = _mm256_set1_ps(s.aux_hid[cc]);
          acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(pr + 0), vh));
          acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(pr + 8), vh));
          acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(pr + 16), vh));
          acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(pr + 24), vh));
        }
        _mm256_storeu_ps(s.hidC + 0, acc0);
        _mm256_storeu_ps(s.hidC + 8, acc1);
        _mm256_storeu_ps(s.hidC + 16, acc2);
        _mm256_storeu_ps(s.hidC + 24, acc3);
      }
      for (int r = 0; r < Head::nhid; ++r) x[Head::kHid + r] = s.hidC[r];
    } else {
      for (int r = 0; r < Head::nhid; ++r) x[Head::kHid + r] = s.aux_hid[r];
    }
  } else {
    for (int r = 0; r < Head::nhid; ++r) x[Head::kHid + r] = 0.0f;
  }

  s.Forward();

  s.skip = override_active ? 1 : 0;
  if (override_active) {
    return base_p;
  }

  float d = s.delta;
  if (d > s.clamp) d = s.clamp;
  else if (d < -s.clamp) d = -s.clamp;
  const float pnew = Sigmoidf(t + d);
  unsigned int q = (unsigned int)(1.0f + 65534.0f * pnew);
  if (q < 1) q = 1;
  if (q > 65535) q = 65535;
  return q;
}

__attribute__((noinline))
void Observe(int bit) {
  Head& s = g_head;
  ++s.nbuf;
  if (s.nbuf >= s.wlen) {
    s.nbuf = 0;
  }
  s.prev_bit = (float)bit;
  s.i = s.i + 1;
}

}  // namespace bit_head
