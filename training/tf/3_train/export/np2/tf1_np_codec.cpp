// Modified in November 2026 by Oleksandr Dreval (https://github.com/OleksandrDreval)
// for cmix-neif-3.1 (https://github.com/OleksandrDreval/cmix-neif-3.1).
// SPDX-License-Identifier: GPL-3.0-only
// tf1_np_codec: writer of the transformer weight container in
// format NP (magic FX2TFWN5/FX2TFWN6) and NP2 (FX2TFWP5/FX2TFWP6;
// reader and layout: the head comment of
// src/tf1/weights_io_compressed.cpp).  A build-time tool of the container,
// like the converters convert_container_v4.py/v5.py/v5k.py of the training
// package; nothing of it is in the compressor.
//
// The order of the hidden units of the exact groups (MLP units of the layers
// without fast weights, the 64 units of the KDA gate bottlenecks, the value
// dims of each head of the plain attention layers) is free: the engine
// computes them with integer dots and per-unit quantisation.  The writer
// chooses that order so that its digits in the factorial number system
// (Laisant 1888, the Lehmer code) spell the first K bytes of the second coder
// stream, which are then not stored (the "bits back" argument of Hinton and
// van Camp 1993).
//
// Origin: the range encoder and the bit trees below are a C++ transcription of
// the encoder of pysrc/weights_compress.py of fx2-cmix-transformer by Vladimir
// Ivanov (GPL-3.0; its range coder follows the LZMA range coder of Igor
// Pavlov), with the models of formats v3 to v5 of this entry (training package
// tf/4_export/weights_compress_v3.py) and v6 (a converter of the development
// tree).  The reader side (models, trees, rope table, canonical names) is
// the reader of this tree, included below.
//
// With one K/V head (MQA, cmix-neif-2.0 form) the value dims of
// a plain attention layer are ONE group of 64 whose order also moves the three
// 64-column blocks of output_projection and the columns of the layer's
// value-embedding table.  Option np2 writes format NP2 (magic
// FX2TFWP5/FX2TFWP6) with the KDA key and value dims as further groups;
// option dead zeroes the parameters of dead KDA key dims (see main()).
//
// Build (from src/cmix-neifv9; all format switches of the reader on):
//   g++ -O2 -std=c++17 -DNEIF_TF1_WC4=1 -DNEIF_TF1_WC5=1 -DNEIF_TF1_WC5K=1 \
//       -DNEIF_TF1_QMIX=1 -DNEIF_TF1_NP2=1 tools_src/tf1_np_codec.cpp \
//       src/tf1/weights_io.cpp -o tf1_np_codec
//
// Usage:
//   tf1_np_codec np <in.c5|in.c6> <out> [k|nok] [fw] [np2] [dead|deadmark]
//       in: a v5 or v6 container (FX2TFWC5/FX2TFWC6, with or without
//       ENC_F32HC).  Steps, each one checked (as in convert_container_v5.py):
//         1. decode in with the reader of this tree;
//         2. identity probe: the plain v5/v6 writer of this tool reproduces
//            in bit for bit (with or without ENC_F32HC, whichever matches);
//         3. write the NP container (k: the bf16-rounded f32 tensors as
//            ENC_F32HC, read only by a build with NEIF_TF1_WC5K=1; nok: as
//            ENC_F32HI; default: as in the input);
//         4. read the NP container back with the reader of this tree: every
//            tensor equals the permuted one, the permutation is not trivial.
//       fw: also the MLP units of the four fast-weight layers.  The fast-weight
//       head reads the unit index (top-k ties, float sums in index order), so
//       the archive then follows the stored order: only for weights without a
//       reference archive.
//       np2: format NP2, also the key dims of each KDA head (key
//       query_convolution, with the q/k projection rows, the k convolution,
//       the forget-gate down rows, their row scales and dt_bias) and the value
//       dims of each KDA layer (one order for the three heads, key block 0 of
//       value_projection, with the v convolution, the output-gate down rows,
//       the output projection columns and the gated output norm).  The engine
//       sums over these dims elementwise or per lane: another order changes
//       its results at the rounding level, the archive follows the order;
//       read only by a build with NEIF_TF1_NP2=1.
//       dead: a KDA key dim whose q, k and forget-gate down rows are all zero
//       never enters the state (k = 0) nor the output (q = 0); its k and q
//       convolution rows and dt_bias are set to zero; with np2 the q
//       convolution is the key of its group, so the further dead dims of a
//       head get distinct marker rows (zero_dead_key_dims()); deadmark: the
//       markers also without np2 (engine probe).  Engine results stay
//       bit-identical; the container is no longer lossless against in.
//   tf1_np_codec ident <container>
//       step 1 and 2 only
//   tf1_np_codec plain <container> <out> [nok]
//       the decoded tensors of any container this reader reads (also NP and
//       NP2) as a plain v5/v6 container (ENC_F32HC; nok: ENC_F32HI), checked
//       by reading it back
//   tf1_np_codec dump <container> <out.raw>
//       the decoded tensors, raw in file order (FX2TFW01 layout)
#include "../src/tf1/weights_io_compressed.cpp"

#include <algorithm>
#include <cstdarg>
#include <map>
#include <string>
#include <vector>

namespace fx2 {
namespace {

enum : uint8_t { ENC_INT4Q = 9 };
constexpr uint32_t CNT_LIMIT_QALPHA = 511;

[[noreturn]] void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
  std::exit(1);
}

size_t dtype_size(uint8_t dt) {
  static const uint8_t kSz[4] = {1, 2, 4, 4};
  return dt < 4 ? kSz[dt] : 1;
}

int canon_name(const std::string& name, std::string* cn) {
  if (name.compare(0, 7, "blocks.") == 0) {
    size_t dot = name.find('.', 7);
    if (dot != std::string::npos) {
      int blk = std::atoi(name.substr(7, dot - 7).c_str());
      *cn = "blocks.#." + name.substr(dot + 1);
      return blk;
    }
  }
  *cn = name;
  return -1;
}

struct Models {
  bool v5 = true;
  std::vector<CountNode> cplane;
  std::vector<CountNode> cint4{16};
  std::vector<CountNode> cint4w;
  std::vector<CountNode> cint4q7;
  std::vector<CountNode> cint4q3;
  std::vector<uint16_t> f32hl;
  CountNode zrow{};
  CountNode qa0[3]{}, qa1[3]{};
  std::vector<uint16_t> bf16_hi = std::vector<uint16_t>(256, 1024);
  std::vector<uint16_t> bf16_lo = std::vector<uint16_t>(256 * 256, 1024);
  std::vector<uint16_t> plane = std::vector<uint16_t>(4 * 256, 1024);
  std::vector<uint16_t> raw = std::vector<uint16_t>(256, 1024);
  std::vector<CountNode> cmeta{256 * 256};
  uint8_t meta_prev = 0;
  CountNode tname{}, tblock{}, tmeta{};
  std::map<uint32_t, std::vector<CountNode>> cname_map;
  CountNode* cname_tree(uint32_t ctx) {
    auto& v = cname_map[ctx];
    if (v.empty()) v.resize(256);
    return v.data();
  }
};

// ------------------------------------------------------------------ tensors
struct Ent {
  std::string name;
  uint8_t dt;
  std::vector<uint32_t> shape;
  std::vector<uint8_t> data;
  size_t numel() const {
    size_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  size_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
  size_t cols() const { return rows() ? numel() / rows() : 0; }
};

// the tensors of a container, decoded by the reader of this tree, in file order
std::vector<Ent> load_ents(const char* path) {
  FILE* fp = std::fopen(path, "rb");
  if (!fp) die("cannot open %s", path);
  char magic[8];
  if (std::fread(magic, 1, 8, fp) != 8) die("short read %s", path);
  if (!std::memcmp(magic, "FX2TFW01", 8)) {
    uint32_t n = 0;
    if (std::fread(&n, 4, 1, fp) != 1) die("short read %s", path);
    std::vector<Ent> out(n);
    for (uint32_t i = 0; i < n; i++) {
      uint32_t nl = 0, nd = 0;
      if (std::fread(&nl, 4, 1, fp) != 1) die("short read %s", path);
      out[i].name.resize(nl);
      if (std::fread(&out[i].name[0], 1, nl, fp) != nl) die("short read %s", path);
      if (std::fread(&out[i].dt, 1, 1, fp) != 1) die("short read %s", path);
      if (std::fread(&nd, 4, 1, fp) != 1) die("short read %s", path);
      out[i].shape.resize(nd);
      if (std::fread(out[i].shape.data(), 4, nd, fp) != nd) die("short read %s", path);
      size_t nb = out[i].numel() * dtype_size(out[i].dt);
      out[i].data.resize(nb);
      if (std::fread(out[i].data.data(), 1, nb, fp) != nb) die("short read %s", path);
    }
    std::fclose(fp);
    return out;
  }
  std::fclose(fp);
  WeightsFile wf = WeightsFile::load_compressed(path);
  std::vector<Ent> out;
  auto push_ci = [&](int ci, int blk) {
    const WTensor& t = wf.T(ci, blk);
    if (!t.data) return;
    Ent x;
    std::string cn = namen::kNamen[ci];
    if (blk >= 0) {
      size_t p = cn.find('#');
      x.name = cn.substr(0, p) + std::to_string(blk) + cn.substr(p + 1);
    } else {
      x.name = cn;
    }
    x.dt = t.dtype;
    x.shape.assign(t.shape, t.shape + t.ndim);
    size_t bytes = t.numel * dtype_size(t.dtype);
    x.data.assign(t.data, t.data + bytes);
    out.push_back(std::move(x));
  };
  for (int ci = 0; ci <= 22; ci++) push_ci(ci, -1);
  for (int blk = 0; blk < 16; blk++) {
    if (blk % 4 != 3) {
      for (int ci = 23; ci <= 61; ci++) push_ci(ci, blk);
    } else {
      for (int ci = 25; ci <= 35; ci++) push_ci(ci, blk);
      for (int ci = 62; ci <= 64; ci++) push_ci(ci, blk);
      for (int ci = 51; ci <= 53; ci++) push_ci(ci, blk);
      push_ci(65, blk);
      for (int ci = 54; ci <= 61; ci++) push_ci(ci, blk);
      if (blk == 15) push_ci(66, blk);
    }
  }
  for (int ci = 67; ci < namen::kN; ci++) push_ci(ci, -1);
  return out;
}

// ------------------------------------------------------------------ encoder
// the range encoder of weights_compress_v3.py (LZMA style, 11-bit nodes)
struct BinEncoder {
  uint64_t low = 0;
  uint32_t range = 0xFFFFFFFFu;
  uint8_t cache = 0;
  uint64_t cache_size = 1;
  std::vector<uint8_t> out;

  void shift_low() {
    if (low < 0xFF000000ull || low > 0xFFFFFFFFull) {
      uint8_t carry = uint8_t(low >> 32);
      out.push_back(uint8_t(cache + carry));
      for (; cache_size > 1; cache_size--) out.push_back(uint8_t(0xFF + carry));
      cache = uint8_t((low >> 24) & 0xFF);
      cache_size = 0;
    }
    cache_size++;
    low = (low << 8) & 0xFFFFFFFFull;
  }
  void norm() {
    while (range < (1u << 24)) {
      range <<= 8;
      shift_low();
    }
  }
  void encode_p(uint32_t p, int bit) {
    if (p < 1) p = 1;
    else if (p > 2047) p = 2047;
    uint32_t bound = (range >> 11) * p;
    if (!bit) {
      range = bound;
    } else {
      low += bound;
      range -= bound;
    }
    norm();
  }
  void encode_bit(uint16_t* prob, int bit, int sh = 5) {
    encode_p(*prob, bit);
    Upd16(prob, bit, sh);
  }
  void encode_tree(uint16_t* probs, int nbits, uint32_t sym, int s0 = 5, int s1 = 5) {
    uint32_t node = 1;
    for (int k = nbits - 1; k >= 0; k--) {
      int bit = (sym >> k) & 1;
      encode_bit(&probs[node], bit, node < 4 ? s0 : s1);
      node = (node << 1) | bit;
    }
  }
  void encode_cbit(CountNode* n, int bit, uint32_t limit) {
    encode_p(ProbCN(n), bit);
    UpdCN(n, bit, limit);
  }
  void encode_ctree(CountNode* nodes, int nbits, uint32_t sym, uint32_t limit) {
    uint32_t node = 1;
    for (int k = nbits - 1; k >= 0; k--) {
      int bit = (sym >> k) & 1;
      encode_cbit(&nodes[node], bit, limit);
      node = (node << 1) | bit;
    }
  }
  std::vector<uint8_t> finish() {
    for (int i = 0; i < 5; i++) shift_low();
    return out;
  }
};

bool ends_with(const std::string& s, const char* t) {
  size_t n = std::strlen(t);
  return s.size() >= n && s.compare(s.size() - n, n, t) == 0;
}

// encoding of a tensor: _pick_encoding + _pick_encoding_v5 of the python
// writer; ver 6 adds ENC_INT4Q (convert_container_v6.py), k ENC_F32HC
uint8_t pick_encoding(const Ent& e, const std::map<std::string, const Ent*>& seen, int ver,
                      bool k) {
  uint8_t enc = ENC_RAW;
  if (e.dt == DT_I8) {
    bool ok = true;
    for (uint8_t b : e.data) {
      int v = int8_t(b);
      if (v < -7 || v > 7) ok = false;
    }
    enc = ok ? ENC_INT4 : ENC_RAW;
  } else if (e.dt == DT_BF16) {
    enc = ENC_BF16;
  } else if (e.dt == DT_F32) {
    enc = ENC_PLANE4;
    if ((e.name == "rope.sin" || e.name == "rope.cos") && e.shape.size() == 2) {
      auto it = seen.find("rope.inv_freq");
      if (it != seen.end() && it->second->numel() == e.shape[1]) {
        const float* inv = reinterpret_cast<const float*>(it->second->data.data());
        int cb = e.name == "rope.cos" ? 1 : 0;
        bool eq = true;
        const float* d = reinterpret_cast<const float*>(e.data.data());
        for (size_t pos = 0; pos < e.shape[0] && eq; pos++)
          for (size_t c = 0; c < e.shape[1]; c++) {
            float v = sincosf_cuda(float(pos) * inv[c], cb);
            if (std::memcmp(&v, &d[pos * e.shape[1] + c], 4)) {
              eq = false;
              break;
            }
          }
        if (eq) enc = cb ? ENC_ROPE_COS : ENC_ROPE_SIN;
      }
    }
  } else if (e.dt == DT_I32) {
    enc = ENC_PLANE4;
  }
  if (enc == ENC_INT4) {
    if (ver >= 6) {
      size_t coarse = 0, R = e.rows(), C = e.cols();
      for (size_t r = 0; r < R; r++) {
        int m = 0;
        for (size_t c = 0; c < C; c++) m = std::max(m, std::abs(int(int8_t(e.data[r * C + c]))));
        if (m > 0 && m <= 3) coarse++;
      }
      if (coarse >= 16) return ENC_INT4Q;
    }
    return ENC_INT4W;
  }
  if (enc == ENC_PLANE4 && e.dt == DT_F32 && e.numel()) {
    bool low0 = true;
    const uint32_t* u = reinterpret_cast<const uint32_t*>(e.data.data());
    for (size_t i = 0; i < e.numel(); i++)
      if (u[i] & 0xFFFF) {
        low0 = false;
        break;
      }
    if (low0) return k ? ENC_F32HC : ENC_F32HI;
  }
  if (enc == ENC_BF16 && ends_with(e.name, ".weight.scale")) return ENC_BF16C;
  return enc;
}

// the value models of one stream (NP: stream A and stream B have one each)
struct ValModels {
  Models m{true};
  std::vector<uint8_t> last_zrow;
  std::string last_int4_cn, last_bf16_cn, last_f32_cn;
  ValModels() {
    m.cplane.resize(2 * 256);
    m.cint4w.resize(16 * 8);
    m.f32hl.assign(257 * 256, 1024);
    bf_hi_prev.resize(256 * 256);
    f32_hi_prev.resize(256 * 256);
    m.cint4q7.resize(24 * 8);
    m.cint4q3.resize(24 * 4);
  }
  std::map<std::string, std::vector<CountNode>> zr_ctx;         // per canonical name, 2
  std::map<std::string, std::vector<CountNode>> cint4w_cn;      // per canonical name, 16 * 8
  std::map<std::string, std::vector<CountNode>> c2d_cn;         // per canonical name, 128 * 8
  std::map<std::string, std::vector<CountNode>> c3d_cn;         // per canonical name, 128 * 8
  std::map<std::string, std::vector<CountNode>> sign_int4w_cn;  // per canonical name, 128
  std::map<std::string, std::vector<CountNode>> sign_2d_cn;     // per canonical name, 256
  std::map<std::string, std::vector<CountNode>> sign_rc_cn;     // per canonical name, 256
  std::map<std::string, std::vector<uint16_t>> bf16c;           // per canonical name, 5 * 257 * 256
  std::map<std::string, std::vector<uint16_t>> f32hl_ci;        // per canonical name, 6 * 257 * 256
  std::map<std::string, std::vector<CountNode>> bf16c_hi;       // per canonical name, 256 hi nodes
  std::map<std::string, std::vector<CountNode>> f32hc;          // per canonical name, 5 hi trees
  std::vector<CountNode> bf_hi_prev;
  std::vector<CountNode> f32_hi_prev;
};

void encode_values(BinEncoder& E, ValModels& vm, const Ent& e, uint8_t enc,
                   const std::string& cn, uint8_t npkey = 0) {
  Models& m = vm.m;
  switch (enc) {
    case ENC_INT4W:
    case ENC_INT4Q: {
      static const uint8_t kU7[5] = {0, 2, 5, 7, 9};
      static const uint8_t kU3[3] = {0, 7, 14};
      static constexpr uint8_t kW2d[4] = {0, 4, 2, 2};
      size_t R = e.rows(), C = e.cols();
      if (enc == ENC_INT4W) vm.last_zrow.assign(R, 0);
      else vm.last_zrow.clear();
      auto& zc_vec = vm.zr_ctx[cn];
      if (zc_vec.empty()) zc_vec.resize(2);
      auto& cw = vm.cint4w_cn[cn];
      auto& c2 = vm.c2d_cn[cn];
      auto& c3 = vm.c3d_cn[cn];
      auto& sw = vm.sign_int4w_cn[cn];
      auto& s2 = vm.sign_2d_cn[cn];
      auto& sr = vm.sign_rc_cn[cn];
      if (cw.empty()) {
        cw.resize(16 * 4);
        c2.resize(64 * 4);
        c3.resize(128 * 4);
        sw.resize(128);
        s2.resize(256);
        sr.resize(256);
        if (!vm.last_int4_cn.empty()) {
          CopyWarmCN(cw.data() + 1, vm.cint4w_cn[vm.last_int4_cn].data() + 1, 16 * 4 - 1, 4);
          CopyWarmCN(c2.data() + 1, vm.c2d_cn[vm.last_int4_cn].data() + 1, 64 * 4 - 1, 4);
          CopyWarmCN(c3.data() + 1, vm.c3d_cn[vm.last_int4_cn].data() + 1, 128 * 4 - 1, 4);
          CopyWarmCN(sw.data(), vm.sign_int4w_cn[vm.last_int4_cn].data(), 128, 4);
          CopyWarmCN(s2.data(), vm.sign_2d_cn[vm.last_int4_cn].data(), 256, 4);
          CopyWarmCN(sr.data(), vm.sign_rc_cn[vm.last_int4_cn].data(), 256, 4);
        }
      } else if (enc == ENC_INT4W) {
        CopyWarmCN(cw.data() + 1, cw.data() + 1, 16 * 4 - 1, 4);
        CopyWarmCN(c2.data() + 1, c2.data() + 1, 64 * 4 - 1, 4);
        CopyWarmCN(c3.data() + 1, c3.data() + 1, 128 * 4 - 1, 4);
      }
      if (enc == ENC_INT4W) vm.last_int4_cn = cn;
      std::vector<uint64_t> cs(C, 0);
      std::vector<int> cstate(C, 0);
      std::vector<uint32_t> c_vis(4 * C, 0), c_n0(4 * C, 0);
      std::vector<uint32_t> c_pos(C, 0);
      uint64_t t_vis[4] = {0}, t_n0[4] = {0};
      uint64_t t_pos = 0, tot_sq = 0;
      uint64_t n = 0, T = 0, N = 0;
      int prev = 0, prev_z = 0;
      for (size_t r = 0; r < R; r++) {
        const int8_t* row = reinterpret_cast<const int8_t*>(&e.data[r * C]);
        bool zero = true;
        for (size_t c = 0; c < C; c++)
          if (row[c]) {
            zero = false;
            break;
          }
        if (enc == ENC_INT4W) {
          CountNode* zc = &zc_vec[prev_z];
          uint32_t pz = (4u * ProbCN(&m.zrow) + 12u * ProbCN(zc)) >> 4;
          int zbit = zero ? 1 : 0;
          E.encode_p(pz, zbit);
          UpdCN(&m.zrow, zbit, CNT_LIMIT_ZROW);
          UpdCN(zc, zbit, CNT_LIMIT_ZROW);
          prev_z = zbit;
        } else {
          E.encode_cbit(&m.zrow, zero, CNT_LIMIT_ZROW);
        }
        if (zero) {
          if (enc == ENC_INT4W) vm.last_zrow[r] = 1;
          continue;
        }
        int alpha = 0;
        if (enc == ENC_INT4Q) {
          int mx = 0;
          for (size_t c = 0; c < C; c++) mx = std::max(mx, std::abs(int(row[c])));
          alpha = mx > 3 ? 0 : (mx > 1 ? 1 : 2);
          E.encode_cbit(&m.qa0[prev], alpha != 0, CNT_LIMIT_QALPHA);
          if (alpha) E.encode_cbit(&m.qa1[prev], alpha - 1, CNT_LIMIT_QALPHA);
          prev = alpha;
        }
        uint64_t rs = 0, r_sq = 0;
        uint32_t ps = 0;
        int r_sbal = 0;
        uint32_t r_vis[4] = {0}, r_n0[4] = {0};
        uint32_t r_pos = 0;
        uint32_t prev_mag = 0, prev2_mag = 0;
        for (size_t k = 0; k < C; k++) {
          int v = row[k];
          uint64_t a = uint64_t(v < 0 ? -v : v);
          if (enc == ENC_INT4W) {
            uint64_t M = (T * 65536 + 16 * 65536) / (N + 16);
            if (M < 1) M = 1;
            uint64_t rr = (rs * 65536 + 6 * M) / (k + 6);
            uint64_t cc = (cs[k] * 65536 + 8 * M) / (n + 8);
            uint64_t w_mag = (rr * cc / M) >> 4;
            uint32_t up_mag = (r > 0) ? uint32_t(std::abs(int(row[k - C]))) : 0;
            uint32_t dg_mag = (r > 0 && k > 0) ? uint32_t(std::abs(int(row[k - C - 1]))) : 0;
            uint32_t ur_mag = (r > 0 && k + 1 < C) ? uint32_t(std::abs(int(row[k - C + 1]))) : 0;
            uint32_t pm_bin = prev_mag == 0 ? 0 : (prev_mag <= 2 ? 1 : (prev_mag <= 4 ? 2 : 3));
            uint32_t um_bin = up_mag == 0 ? 0 : (up_mag <= 2 ? 1 : (up_mag <= 4 ? 2 : 3));
            uint32_t ctx2d_base = (pm_bin << 2) | um_bin;
            uint32_t ctx3d_base = (pm_bin << 3) | ((prev2_mag > 0 ? 1u : 0u) << 2) |
                                  (((dg_mag + ur_mag) > 0 ? 1u : 0u) << 1) | (up_mag > 0 ? 1u : 0u);
            uint32_t node = 1;
            for (int bit_k = 2; bit_k >= 1; bit_k--) {
              int bit = (uint32_t(a) >> bit_k) & 1;
              constexpr uint64_t dr1 = 384, dc1 = 12, dr2 = 64, dc2 = 20;
              uint64_t t_n1 = t_vis[node] - t_n0[node];
              uint64_t r_n1 = r_vis[node] - r_n0[node];
              uint64_t c_n1 = c_vis[node * C + k] - c_n0[node * C + k];
              uint64_t p1_t = ((t_n1 + 4) << 12) / (t_vis[node] + 8);
              if (p1_t < 1) p1_t = 1;
              uint64_t p1_r = ((r_n1 << 12) + dr1 * p1_t) / (r_vis[node] + dr1);
              uint64_t p1_c = ((c_n1 << 12) + dc1 * p1_t) / (c_vis[node * C + k] + dc1);
              uint64_t w_nd = ((p1_r * p1_c) / p1_t) * 4096ull / p1_t;
              uint64_t w_comb = (w_mag + w_nd) >> 1;
              uint64_t b = (w_comb * 16ull) / 13312;
              if (b >= 16) b = 15;
              uint32_t ctx2d = (uint32_t(b >> 2) << 4) | ctx2d_base;
              uint32_t ctx3d = (uint32_t(b >> 2) << 5) | ctx3d_base;
              CountNode* cn_n = &cw[b * 4 + node];
              CountNode* cn2 = &c2[ctx2d * 4 + node];
              CountNode* cn3 = &c3[ctx3d * 4 + node];
              uint32_t p_tree = ProbCN(cn_n), p_2d = ProbCN(cn2), p_3d = ProbCN(cn3);
              uint64_t p_t = (t_n0[node] * 2048ull + 8ull * p_tree) / (t_vis[node] + 8ull);
              if (p_t < 1) p_t = 1;
              uint64_t p_r = (uint64_t(r_n0[node]) * 2048ull + dr2 * p_t) / (r_vis[node] + dr2);
              uint64_t p_c = (uint64_t(c_n0[node * C + k]) * 2048ull + dc2 * p_t) / (c_vis[node * C + k] + dc2);
              uint64_t p_rc = (p_r * p_c) / p_t;
              if (p_rc > 2047) p_rc = 2047;
              uint64_t wt = 23u - kW2d[node];
              uint32_t p_final = uint32_t((wt * p_tree + kW2d[node] * p_2d + p_3d + 8u * p_rc) >> 5);
              E.encode_p(p_final, bit);
              UpdCN(cn_n, bit, 3071);
              UpdCN(cn2, bit, 3071);
              UpdCN(cn3, bit, 3071);
              r_vis[node]++; c_vis[node * C + k]++; t_vis[node]++;
              if (!bit) { r_n0[node]++; c_n0[node * C + k]++; t_n0[node]++; }
              node = (node << 1) | uint32_t(bit);
            }
            node = (node << 1) | uint32_t(node == 7);
            prev2_mag = prev_mag;
            prev_mag = uint32_t(a);
            if (a) {
              int bit = v < 0 ? 1 : 0;
              int st = cstate[k];
              uint32_t ctx = (ps & 3u) | (((uint32_t(a) - 1u) >> 1) << 2) | (uint32_t(st & 3) << 4) | (uint32_t(st < 0) << 6);
              int up_v = (r > 0) ? int(row[k - C]) : 0;
              int left_v = (k > 0) ? int(row[k - 1]) : 0;
              uint32_t up_s = up_v == 0 ? 0u : (up_v < 0 ? 2u : 1u);
              uint32_t lf_s = left_v == 0 ? 0u : (left_v < 0 ? 2u : 1u);
              int c_sum = st >> 2;
              uint32_t c_str = c_sum == 0 ? 0u : (c_sum < -12 ? 3u : (c_sum < 0 ? 2u : 1u));
              uint32_t ctx2 = (lf_s & 3u) | ((up_s & 3u) << 2) | ((c_str & 3u) << 4) | (((uint32_t(a) - 1u) >> 1) << 6);
              uint32_t r_pol = r_sbal == 0 ? 0u : (r_sbal < -8 ? 3u : (r_sbal < 0 ? 2u : 1u));
              uint32_t ctx3 = (ps & 3u) | ((r_pol & 3u) << 2) | ((c_str & 3u) << 4) | ((up_s & 3u) << 6);
              uint32_t p1 = ProbCN(&sw[ctx]), p2 = ProbCN(&s2[ctx2]), p3 = ProbCN(&sr[ctx3]);
              uint64_t ps_t = (t_pos * 2048ull + 64ull * 1024ull) / (T + 64ull);
              if (ps_t < 1) ps_t = 1;
              uint64_t ps_r = (uint64_t(r_pos) * 2048ull + 192ull * ps_t) / (rs + 192ull);
              uint64_t ps_c = (uint64_t(c_pos[k]) * 2048ull + 192ull * ps_t) / (cs[k] + 192ull);
              uint64_t p_src = (ps_r * ps_c) / ps_t;
              if (p_src > 2047) p_src = 2047;
              uint32_t psign = (11u * p1 + 6u * p2 + 4u * p3 + 11u * uint32_t(p_src)) >> 5;
              E.encode_p(psign, bit);
              UpdCN(&sw[ctx], bit, CNT_LIMIT_SIGN);
              UpdCN(&s2[ctx2], bit, CNT_LIMIT_SIGN);
              UpdCN(&sr[ctx3], bit, CNT_LIMIT_SIGN);
              ps = (ps << 1) | uint32_t(bit);
              cstate[k] = ((st & ~3) + (v << 2)) | (bit + 1);
              r_sbal += v;
              if (!bit) { r_pos += uint32_t(a); c_pos[k] += uint32_t(a); t_pos += uint32_t(a); }
            }
          } else {
            uint64_t M = (T * 65536 + 64 * 190054) / (N + 64);
            if (M < 1) M = 1;
            uint64_t rr = (rs * 65536 + 64 * M) / (k + 64);
            uint64_t cc = (cs[k] * 65536 + 64 * M) / (n + 64);
            uint64_t b = (8 * rr * cc) / (M * 65536);
            b = b < 8 ? 0 : (b > 8 + 23 ? 23 : b - 8);
            if (alpha == 0) E.encode_ctree(&m.cint4w[b * 16], 4, v + 7, CNT_LIMIT_INT4W);
            else if (alpha == 1) E.encode_ctree(&m.cint4q7[b * 8], 3, v + 3, CNT_LIMIT_INT4W);
            else E.encode_ctree(&m.cint4q3[b * 4], 2, v + 1, CNT_LIMIT_INT4W);
            if (alpha == 1) a = kU7[a];
            else if (alpha == 2) a = kU3[a];
          }
          rs += a;
          cs[k] += a;
          T += a;
          N++;
          r_sq += a * a;
        }
        n++;
        if (enc == ENC_INT4W) {
          tot_sq += r_sq;
          uint64_t mean_sq = tot_sq / n;
          if (mean_sq < 1) mean_sq = 1;
          int rb = (r_sq * 10 < mean_sq * 9) ? 0 : (r_sq < mean_sq ? 1 : (r_sq * 10 < mean_sq * 11 ? 2 : 3));
          vm.last_zrow[r] = uint8_t(rb << 1);
        }
      }
      break;
    }
    case ENC_F32HI:
    case ENC_F32HC:
    case ENC_BF16C: {
      bool f32 = enc != ENC_BF16C;
      std::vector<uint16_t>* s = &m.f32hl;
      CountNode* bhi = nullptr;
      if (!f32) {
        auto& t = vm.bf16c[cn];
        if (t.empty()) {
          t.assign(5 * 257 * 256, 1024);
          if (!vm.last_bf16_cn.empty()) {
            auto& pt = vm.bf16c[vm.last_bf16_cn];
            for (size_t u = 0; u < 5 * 257 * 256; u++) t[u] = uint16_t((pt[u] + 1024) >> 1);
          }
        }
        auto& th = vm.bf16c_hi[cn];
        if (th.empty()) {
          th.resize(256);
          if (!vm.last_bf16_cn.empty()) {
            auto& pth = vm.bf16c_hi[vm.last_bf16_cn];
            for (size_t u = 0; u < 256; u++) {
              uint16_t g0 = pth[u].n0 >> 4, g1 = pth[u].n1 >> 4;
              th[u] = {uint16_t(g0 ? g0 : 1), uint16_t(g1 ? g1 : 1)};
            }
          }
        }
        vm.last_bf16_cn = cn;
        s = &t;
        bhi = th.data();
      }
      CountNode* hc = nullptr;
      if (enc == ENC_F32HC) {
        auto& c = vm.f32hc[cn];
        auto& fc = vm.f32hl_ci[cn];
        if (c.empty()) {
          c.resize(5 * 256);
          fc.assign(6 * 257 * 256, 1024);
          if (!vm.last_f32_cn.empty()) {
            auto& pc = vm.f32hc[vm.last_f32_cn];
            for (size_t u = 0; u < 5 * 256; u++) {
              uint16_t g0 = pc[u].n0 >> 4, g1 = pc[u].n1 >> 4;
              c[u] = {uint16_t(g0 ? g0 : 1), uint16_t(g1 ? g1 : 1)};
            }
            auto& pfc = vm.f32hl_ci[vm.last_f32_cn];
            for (size_t u = 0; u < 6 * 257 * 256; u++) fc[u] = uint16_t((pfc[u] + 1024) >> 1);
          }
        }
        vm.last_f32_cn = cn;
        hc = c.data();
        s = &fc;
      }
      const size_t rows = e.rows(), cols = e.cols();
      uint32_t prev_hi = 126;
      uint32_t prev_col_hi[5] = {126, 126, 126, 126, 126};
      for (size_t i = 0, j = 0; i < e.numel(); i++) {
        if (!f32 && i < vm.last_zrow.size() && (vm.last_zrow[i] & 1)) continue;
        if (f32 && !npkey && rows == vm.last_zrow.size() && (vm.last_zrow[i / cols] & 1)) continue;
        uint32_t v;
        if (f32) {
          uint32_t u;
          std::memcpy(&u, &e.data[4 * i], 4);
          v = u >> 16;
        } else {
          uint16_t u;
          std::memcpy(&u, &e.data[2 * i], 2);
          v = u;
        }
        uint32_t hi = v >> 8, lo = v & 0xFF;
        if (hc || bhi) {
          size_t jb = f32 ? (j < 4 ? j : 4) : ((i < vm.last_zrow.size()) ? (vm.last_zrow[i] >> 1) : 0);
          uint32_t phi = (f32 && cols == 4) ? prev_col_hi[jb] : prev_hi;
          CountNode* hc1 = f32 ? (hc + jb * 256) : bhi;
          CountNode* hc2 = (f32 ? vm.f32_hi_prev.data() : vm.bf_hi_prev.data()) + phi * 256;
          uint32_t w_phi = (f32 && cols != 4) ? 3u : 4u;
          uint32_t lim = f32 ? CNT_LIMIT_F32HC : CNT_LIMIT_BF16C_HI;
          uint32_t node = 1;
          for (int bk = 7; bk >= 0; bk--) {
            int bit = (hi >> bk) & 1;
            uint32_t p = ((16u - w_phi) * ProbCN(&hc1[node]) + w_phi * ProbCN(&hc2[node])) >> 4;
            E.encode_p(p, bit);
            UpdCN(&hc1[node], bit, lim);
            UpdCN(&hc2[node], bit, 511);
            node = (node << 1) | uint32_t(bit);
          }
          if (f32) {
            prev_col_hi[jb] = hi;
            if (++j == cols) j = 0;
          }
          prev_hi = hi;
          uint16_t* s_ci = &(*s)[256 + hi * 256];
          uint16_t* s_sub = s_ci + (jb + 1) * 257 * 256;
          int s0 = f32 ? (hi ? 5 : 2) : 4;
          int s1 = f32 ? (hi ? 8 : 2) : 5;
          int nbits = 4;
          node = 1;
          for (int bk = 7; bk >= 8 - nbits; bk--) {
            int bit = (lo >> bk) & 1;
            uint32_t p = f32 ? ((3u * s_ci[node] + 10u * s_sub[node] + 3u * m.f32hl[256 + hi * 256 + node]) >> 4)
                             : ((10u * s_ci[node] + 6u * s_sub[node]) >> 4);
            E.encode_p(p, bit);
            int sh = node < 4 ? s0 : s1;
            Upd16(&s_ci[node], bit, sh);
            Upd16(&s_sub[node], bit, sh);
            if (f32) Upd16(&m.f32hl[256 + hi * 256 + node], bit, sh);
            node = (node << 1) | uint32_t(bit);
          }
        } else {
          E.encode_tree(s->data(), 8, hi);
          if (++j == cols) j = 0;
          E.encode_tree(&(*s)[256 + hi * 256], 8, lo, f32 ? (hi ? 5 : 2) : 4, f32 ? (hi ? 8 : 2) : 5);
        }
      }
      break;
    }
    case ENC_BF16: {
      for (size_t i = 0; i < e.numel(); i++) {
        uint16_t u;
        std::memcpy(&u, &e.data[2 * i], 2);
        uint32_t hi = u >> 8;
        E.encode_tree(m.bf16_hi.data(), 8, hi);
        E.encode_tree(&m.bf16_lo[hi * 256], 8, u & 0xFF);
      }
      break;
    }
    case ENC_PLANE4: {
      for (size_t i = 0; i < e.data.size(); i++) {
        if ((i & 3) < 2) E.encode_ctree(&m.cplane[(i & 3) * 256], 8, e.data[i], CNT_LIMIT_PLANE01);
        else E.encode_tree(&m.plane[(i & 3) * 256], 8, e.data[i]);
      }
      break;
    }
    case ENC_ROPE_SIN:
    case ENC_ROPE_COS:
      break;  // recomputed by the reader
    case ENC_RAW:
      for (uint8_t b : e.data) E.encode_tree(m.raw.data(), 8, b);
      break;
    default:
      die("encoder: encoding %u", unsigned(enc));
  }
}

// name and metadata through the v4 template
struct HeadCoder {
  Models m{true};
  std::vector<CountNode> cm_role{8 * 256};
  std::vector<CountNode> cn1{256 * 256};
  std::vector<CountNode> cn0{256};
  std::map<std::string, std::string> succ;
  std::map<std::string, std::vector<uint8_t>> meta;
  std::string prev;
  bool have_prev = false;
  int prev_blk = -1;

  void put_meta(BinEncoder& E, uint8_t b, size_t role) {
    CountNode* t1 = &m.cmeta[size_t(m.meta_prev) * 256];
    CountNode* t2 = &cm_role[role * 256];
    uint32_t w = role ? 15u : 10u;
    uint32_t node = 1;
    for (int k = 7; k >= 0; k--) {
      int bit = (b >> k) & 1;
      uint32_t p = ((16u - w) * ProbCN(&t1[node]) + w * ProbCN(&t2[node])) >> 4;
      E.encode_p(p, bit);
      UpdCN(&t1[node], bit, CNT_LIMIT_META);
      UpdCN(&t2[node], bit, CNT_LIMIT_META);
      node = (node << 1) | uint32_t(bit);
    }
    m.meta_prev = b;
  }
  void put_name(BinEncoder& E, const std::string& name) {
    put_meta(E, uint8_t(name.size()), 0);
    uint32_t c2 = 0, c1 = 0;
    for (unsigned char ch : name) {
      CountNode* tree = m.cname_tree(((c2 & 127u) << 7) | (c1 & 127u));
      CountNode* t1 = &cn1[c1 * 256];
      uint32_t node = 1;
      for (int bk = 7; bk >= 0; bk--) {
        int bit = (ch >> bk) & 1;
        uint32_t p2 = ProbCN(&tree[node]), p1 = ProbCN(&t1[node]), p0 = ProbCN(&cn0[node]);
        uint32_t p = (uint32_t(tree[node].n0) + tree[node].n1 <= 3u)
                         ? ((2u * p2 + 12u * p1 + 2u * p0) >> 4)
                         : ((15u * p2 + p1) >> 4);
        E.encode_p(p, bit);
        UpdCN(&tree[node], bit, CNT_LIMIT_NAME);
        UpdCN(&t1[node], bit, CNT_LIMIT_NAME);
        UpdCN(&cn0[node], bit, CNT_LIMIT_NAME);
        node = (node << 1) | uint32_t(bit);
      }
      c2 = c1;
      c1 = ch;
    }
  }
  // returns the canonical name
  std::string code(BinEncoder& E, const Ent& e, uint8_t enc_byte) {
    std::vector<uint8_t> mt;
    mt.push_back(e.dt);
    mt.push_back(uint8_t(e.shape.size()));
    for (auto d : e.shape)
      for (int k = 0; k < 4; k++) mt.push_back(uint8_t(d >> (8 * k)));
    mt.push_back(enc_byte);
    std::string cn;
    int blk = canon_name(e.name, &cn);
    bool hit;
    std::string pred;
    bool has_pred = false;
    if (have_prev) {
      auto it = succ.find(prev);
      if (it != succ.end()) {
        pred = it->second;
        has_pred = true;
      }
    }
    if (blk >= 0)
      hit = has_pred && pred == cn && (blk == prev_blk || blk == prev_blk + 1);
    else
      hit = has_pred && pred == cn && cn.compare(0, 9, "blocks.#.") != 0;
    E.encode_cbit(&m.tname, hit, CNT_LIMIT_TMPL);
    if (hit) {
      if (blk >= 0) E.encode_cbit(&m.tblock, blk == prev_blk, CNT_LIMIT_TMPL);
    } else {
      put_name(E, e.name);
    }
    auto mi = meta.find(cn);
    bool same = mi != meta.end() && mi->second == mt;
    E.encode_cbit(&m.tmeta, same, CNT_LIMIT_TMPL);
    if (!same) {
      put_meta(E, mt[0], 1);
      put_meta(E, mt[1], 2);
      for (size_t k = 2; k < mt.size(); k++)
        put_meta(E, mt[k], k + 1 == mt.size() ? 7 : (3 + ((k - 2) & 3)));
    }
    if (have_prev) succ[prev] = cn;
    meta[cn] = mt;
    prev = cn;
    have_prev = true;
    if (blk >= 0) prev_blk = blk;
    return cn;
  }
};

std::vector<uint8_t> u32le(uint32_t v) {
  return {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
}

std::vector<uint8_t> encodings(const std::vector<Ent>& es, int ver, bool k) {
  std::vector<uint8_t> enc(es.size());
  std::map<std::string, const Ent*> seen;
  for (size_t i = 0; i < es.size(); i++) {
    enc[i] = pick_encoding(es[i], seen, ver, k);
    if (es[i].dt == DT_F32) seen[es[i].name] = &es[i];
  }
  return enc;
}

// the plain v5/v6 container (one stream)
std::vector<uint8_t> encode_plain(const std::vector<Ent>& es, int ver, bool k) {
  BinEncoder E;
  HeadCoder H;
  ValModels V;
  std::vector<uint8_t> enc = encodings(es, ver, k);
  for (size_t i = 0; i < es.size(); i++) {
    std::string cn = H.code(E, es[i], enc[i]);
    encode_values(E, V, es[i], enc[i], cn);
  }
  const char* magic = ver >= 6 ? "FX2TFWC6" : "FX2TFWC5";
  std::vector<uint8_t> out(magic, magic + 8);
  auto n = u32le(uint32_t(es.size()));
  out.insert(out.end(), n.begin(), n.end());
  auto s = E.finish();
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

// ------------------------------------------------------------------ groups
// a group: the elements are the rows [r0, r0+n) of the key tensor; moving
// element k moves the row o+k of each (tensor, o) in rows_at, the element o+k
// of elem_at (row scales, 1-d tensors) and the column o+k of cols_at.  flag:
// the key flag of the encoding byte (0x40 all rows, 0x20 blocks of 64 rows,
// 0x60 the first 64 rows only, format NP2).
struct Group {
  std::string name;
  int key;
  uint8_t flag;
  int r0, n;
  std::vector<std::pair<int, int>> rows_at, elem_at, cols_at;
};

int find_t(const std::vector<Ent>& es, const std::string& n) {
  for (size_t i = 0; i < es.size(); i++)
    if (es[i].name == n) return int(i);
  die("tensor %s missing", n.c_str());
}

int find_opt(const std::vector<Ent>& es, const std::string& n) {
  for (size_t i = 0; i < es.size(); i++)
    if (es[i].name == n) return int(i);
  return -1;
}

// a group over rows [r0, r0+n) of key that moves rows, scales and columns at
// the same offset r0
Group group_at(const std::string& name, int key, int r0, int n, std::vector<int> rows,
               std::vector<int> elem, std::vector<int> cols, int key_rows) {
  Group G;
  G.name = name;
  G.key = key;
  G.flag = n == key_rows ? 0x40 : 0x20;
  G.r0 = r0;
  G.n = n;
  for (int t : rows) G.rows_at.push_back({t, r0});
  for (int t : elem) G.elem_at.push_back({t, r0});
  for (int t : cols) G.cols_at.push_back({t, r0});
  return G;
}

// the groups: exact ones (format NP); fw: also the MLP units of the last four
// (fast-weight) layers; kda: also the key and value dims of the KDA layers
// (format NP2)
std::vector<Group> np_groups(const std::vector<Ent>& es, bool with_fw, bool kda) {
  const Ent& ck = es[find_t(es, "config.kimi")];
  int NL = int(ck.numel());
  const int32_t* kimi = reinterpret_cast<const int32_t*>(ck.data.data());
  std::vector<Group> gs;
  int vi_count = 0;  // index of the plain attention layer (value_embeddings.<i>)
  for (int l = 0; l < NL; l++) {
    std::string b = "blocks." + std::to_string(l) + ".";
    std::string a = b + "attention.";
    if (kimi[l]) {
      for (const char* g : {"forget_gate_projection", "output_gate_projection"}) {
        std::string p = a + g;
        int key = find_t(es, p + ".up.weight.q");
        int n = int(es[key].rows());
        gs.push_back(group_at(p, key, 0, n, {key}, {find_t(es, p + ".up.weight.scale")},
                              {find_t(es, p + ".down.weight.q")}, n));
      }
      if (kda) {
        // key dims of head h: rows 64h.. of the q/k convolutions and
        // projections and of the forget-gate down projection, their scales
        // and dt_bias (the per-dim decay of the state rows)
        for (int h = 0; h < 3; h++) {
          std::vector<int> rows, elem;
          for (const char* t : {"query_convolution.weight", "key_convolution.weight",
                                "query_projection.weight.q", "key_projection.weight.q",
                                "forget_gate_projection.down.weight.q"})
            rows.push_back(find_t(es, a + t));
          for (const char* t : {"query_projection.weight.scale", "key_projection.weight.scale",
                                "forget_gate_projection.down.weight.scale", "dt_bias"})
            elem.push_back(find_t(es, a + t));
          int key = find_t(es, a + "query_convolution.weight");
          gs.push_back(group_at(a + "key_dims.h" + std::to_string(h), key, 64 * h, 64, rows, elem,
                                {}, int(es[key].rows())));
        }
        // value dims: one order for the three heads (the gated output norm
        // weight is shared by the heads); key block 0 of value_projection
        Group G;
        G.name = a + "value_dims";
        G.key = find_t(es, a + "value_projection.weight.q");
        G.flag = 0x60;
        G.r0 = 0;
        G.n = 64;
        for (int h = 0; h < 3; h++) {
          for (const char* t : {"value_convolution.weight", "value_projection.weight.q",
                                "output_gate_projection.down.weight.q"})
            G.rows_at.push_back({find_t(es, a + t), 64 * h});
          for (const char* t : {"value_projection.weight.scale",
                                "output_gate_projection.down.weight.scale"})
            G.elem_at.push_back({find_t(es, a + t), 64 * h});
          G.cols_at.push_back({find_t(es, a + "output_projection.weight.q"), 64 * h});
        }
        G.elem_at.push_back({find_t(es, a + "output_fused_norm_gate.weight"), 0});
        gs.push_back(G);
      }
    } else if (es[find_t(es, a + "value_projection.weight.q")].rows() == 64) {
      // one K/V head (MQA): its 64 value dims, moved in V, in the three head
      // blocks of the output projection and in the columns of the
      // value-embedding table of this layer
      int key = find_t(es, a + "value_projection.weight.q");
      Group G = group_at(a + "v_dims.mqa", key, 0, 64, {key},
                         {find_t(es, a + "value_projection.weight.scale")}, {}, 64);
      const int op = find_t(es, a + "output_projection.weight.q");
      G.cols_at = {{op, 0}, {op, 64}, {op, 128}};
      int ve = find_opt(es, "value_embeddings." + std::to_string(vi_count) + ".inner.weight.q");
      if (ve >= 0) G.cols_at.push_back({ve, 0});
      gs.push_back(G);
      vi_count++;
    } else {
      for (int h = 0; h < 3; h++) {
        int key = find_t(es, a + "value_projection.weight.q");
        gs.push_back(group_at(a + "v_dims.h" + std::to_string(h), key, 64 * h, 64, {key},
                              {find_t(es, a + "value_projection.weight.scale")},
                              {find_t(es, a + "output_projection.weight.q")}, int(es[key].rows())));
      }
    }
    if (l < NL - 4 || with_fw) {
      int key = find_t(es, b + "mlp.up.weight.q");
      int n = int(es[key].rows());
      gs.push_back(group_at(b + "mlp", key, 0, n, {key}, {find_t(es, b + "mlp.up.weight.scale")},
                            {find_t(es, b + "mlp.down.weight.q")}, n));
    }
  }
  // the order in which the reader meets the key tensors
  std::stable_sort(gs.begin(), gs.end(), [](const Group& x, const Group& y) {
    return x.key != y.key ? x.key < y.key : x.r0 < y.r0;
  });
  return gs;
}

// new element k = old element perm[k]
void apply_perm(std::vector<Ent>& es, const Group& G, const std::vector<int>& perm) {
  for (auto ra : G.rows_at) {
    Ent& e = es[ra.first];
    size_t C = e.cols() * dtype_size(e.dt);
    if (e.shape.size() < 2) die("group %s: rows of the 1-d tensor %s", G.name.c_str(), e.name.c_str());
    std::vector<uint8_t> old(e.data.begin() + size_t(ra.second) * C,
                             e.data.begin() + size_t(ra.second + G.n) * C);
    for (int k = 0; k < G.n; k++)
      std::memcpy(&e.data[size_t(ra.second + k) * C], &old[size_t(perm[k]) * C], C);
  }
  for (auto ea : G.elem_at) {
    Ent& e = es[ea.first];
    size_t s = dtype_size(e.dt);
    std::vector<uint8_t> old(e.data.begin() + size_t(ea.second) * s,
                             e.data.begin() + size_t(ea.second + G.n) * s);
    for (int k = 0; k < G.n; k++)
      std::memcpy(&e.data[size_t(ea.second + k) * s], &old[size_t(perm[k]) * s], s);
  }
  for (auto ca : G.cols_at) {
    Ent& e = es[ca.first];
    size_t R = e.rows(), C = e.cols(), s = dtype_size(e.dt);
    if (size_t(ca.second + G.n) > C) die("group %s: column block outside %s", G.name.c_str(), e.name.c_str());
    for (size_t r = 0; r < R; r++) {
      uint8_t* row = &e.data[r * C * s];
      std::vector<uint8_t> old(row + size_t(ca.second) * s, row + size_t(ca.second + G.n) * s);
      for (int k = 0; k < G.n; k++)
        std::memcpy(row + size_t(ca.second + k) * s, &old[size_t(perm[k]) * s], s);
    }
  }
}

// key rows are compared over their bytes (format NP2: f32 key tensors)
size_t key_bytes(const std::vector<Ent>& es, const Group& G) {
  const Ent& e = es[G.key];
  return e.cols() * dtype_size(e.dt);
}

const uint8_t* key_row(const std::vector<Ent>& es, const Group& G, int k) {
  return &es[G.key].data[size_t(G.r0 + k) * key_bytes(es, G)];
}

// option dead: zero the parameters of the dead KDA key dims (q, k and
// forget-gate down rows all zero).  With distinct (np2: the q convolution is
// the key of the key-dim groups) only the first dead dim of a head gets a
// zero q convolution row; the j-th further one (j = 0, 1, ...) gets the
// marker row with element j mod 4 = the f32 of high half 1 + j / 4 (a
// denormal) and zeros elsewhere: any finite value works there, since its
// input channel is zero.  Returns the number of dead dims; *qmark the marker
// rows.
int zero_dead_key_dims(std::vector<Ent>& es, bool distinct, int* qmark) {
  const Ent& ck = es[find_t(es, "config.kimi")];
  const int32_t* kimi = reinterpret_cast<const int32_t*>(ck.data.data());
  int dead = 0;
  *qmark = 0;
  for (int l = 0; l < int(ck.numel()); l++) {
    if (!kimi[l]) continue;
    std::string a = "blocks." + std::to_string(l) + ".attention.";
    const Ent& q = es[find_t(es, a + "query_projection.weight.q")];
    const Ent& k = es[find_t(es, a + "key_projection.weight.q")];
    const Ent& fd = es[find_t(es, a + "forget_gate_projection.down.weight.q")];
    Ent& qc = es[find_t(es, a + "query_convolution.weight")];
    Ent& kc = es[find_t(es, a + "key_convolution.weight")];
    Ent& dt = es[find_t(es, a + "dt_bias")];
    const size_t cb = qc.cols() * dtype_size(qc.dt), kb = kc.cols() * dtype_size(kc.dt);
    const size_t ds = dtype_size(dt.dt);
    if (qc.dt != DT_F32 || qc.cols() != 4) die("%squery_convolution.weight: f32 [*, 4] expected", a.c_str());
    int extra[3] = {-1, -1, -1};  // per head: further dead dims so far (-1: none dead yet)
    for (size_t r = 0; r < q.rows(); r++) {
      bool z = true;
      for (const Ent* e : {&q, &k, &fd})
        for (size_t c = 0; c < e->cols() && z; c++) z = e->data[r * e->cols() + c] == 0;
      if (!z) continue;
      dead++;
      std::memset(&kc.data[r * kb], 0, kb);
      std::memset(&dt.data[r * ds], 0, ds);
      std::memset(&qc.data[r * cb], 0, cb);
      int& j = extra[r / 64];
      if (distinct && j >= 0) {
        const uint32_t v = uint32_t(1 + j / 4) << 16;
        std::memcpy(&qc.data[r * cb + 4 * size_t(j % 4)], &v, 4);
        (*qmark)++;
      }
      j++;
    }
  }
  return dead;
}

// ------------------------------------------------------------------ bignum
struct Big {
  std::vector<uint32_t> l;  // little-endian limbs
  void mul_add(uint32_t m, uint32_t a) {
    uint64_t c = a;
    for (auto& x : l) {
      uint64_t t = uint64_t(x) * m + c;
      x = uint32_t(t);
      c = t >> 32;
    }
    if (c) l.push_back(uint32_t(c));
  }
  uint32_t divmod(uint32_t d) {  // this /= d, returns the remainder
    uint64_t r = 0;
    for (size_t i = l.size(); i-- > 0;) {
      uint64_t t = (r << 32) | l[i];
      l[i] = uint32_t(t / d);
      r = t % d;
    }
    while (!l.empty() && !l.back()) l.pop_back();
    return uint32_t(r);
  }
  size_t bitlen() const {
    if (l.empty()) return 0;
    size_t b = 32 * (l.size() - 1);
    uint32_t t = l.back();
    while (t) {
      b++;
      t >>= 1;
    }
    return b;
  }
  bool zero() const { return l.empty(); }
};

// whole bytes the groups can carry: floor((bitlen(prod n!) - 1) / 8)
size_t capacity_bytes(const std::vector<Group>& gs) {
  Big P;
  P.l = {1};
  for (const auto& G : gs)
    for (int i = 2; i <= G.n; i++) P.mul_add(uint32_t(i), 0);
  return (P.bitlen() - 1) / 8;
}

// the NP container:
//   magic | u32 n | u32 nA | u32 lenA | u32 K | A[1..] | B[1+K..]
// Stream A = the tensors up to the last one a group touches (file order);
// stream B = the rest.  B continues all models of A except the int4 trees and
// the per-name trees of the row scales (those start fresh), so its bytes do
// not depend on the order of the rows in A.  B[1..K] (B[0] is the coder's zero
// cache byte) are carried by the orders: per group, digit k of n elements has
// radix n - k and is the rank of the stored element among the remaining ones
// in bytewise order of the key row; the reader folds the digits in file order.
// Format NP2 (np2): stream B also starts the f32 models fresh, magic FX2TFWP5/6.
std::vector<uint8_t> encode_np(std::vector<Ent> es, int ver, bool k, const std::vector<Group>& gs,
                               bool np2, size_t* K_out, std::vector<Ent>* permuted_out) {
  std::vector<bool> inA(es.size(), false);
  std::vector<uint8_t> keyflag(es.size(), 0);
  for (const auto& G : gs) {
    for (auto p : G.rows_at) inA[p.first] = true;
    for (auto p : G.elem_at) inA[p.first] = true;
    for (auto p : G.cols_at) inA[p.first] = true;
    inA[G.key] = true;
    if (G.flag != 0x40 && G.flag != 0x20 && !(np2 && G.flag == 0x60))
      die("group %s: key flag 0x%02x", G.name.c_str(), unsigned(G.flag));
    // format NP leaves the f32 models of stream B running: no group may move f32 values
    for (const auto* v : {&G.rows_at, &G.elem_at, &G.cols_at})
      for (auto p : *v)
        if (!np2 && es[p.first].dt == DT_F32)
          die("group %s moves the f32 tensor %s: format NP2 only", G.name.c_str(),
              es[p.first].name.c_str());
    if (keyflag[G.key] && keyflag[G.key] != G.flag)
      die("tensor %s: two key flags", es[G.key].name.c_str());
    keyflag[G.key] = G.flag;
  }
  // encodings of the unpermuted tensors (an order does not change them)
  std::vector<uint8_t> enc = encodings(es, ver, k);
  std::vector<int> ia, ib;
  size_t last = 0;
  for (size_t i = 0; i < es.size(); i++)
    if (inA[i]) last = i;
  for (size_t i = 0; i < es.size(); i++) (i <= last ? ia : ib).push_back(int(i));
  // stream B: the header models after A (values do not enter them) and the
  // value models after A, int4 trees and per-name scale trees fresh
  HeadCoder HB;
  ValModels VB;
  {
    BinEncoder scratch;
    for (int i : ia) {
      std::string cn = HB.code(scratch, es[i], uint8_t(enc[i] | keyflag[i]));
      encode_values(scratch, VB, es[i], enc[i], cn, keyflag[i]);
    }
    Models& m = VB.m;
    std::fill(m.cint4.begin(), m.cint4.end(), CountNode());
    std::fill(m.cint4w.begin(), m.cint4w.end(), CountNode());
    std::fill(m.cint4q7.begin(), m.cint4q7.end(), CountNode());
    std::fill(m.cint4q3.begin(), m.cint4q3.end(), CountNode());
    m.zrow = CountNode();
    VB.zr_ctx.clear();
    VB.sign_int4w_cn.clear();
    VB.sign_2d_cn.clear();
    VB.sign_rc_cn.clear();
    VB.last_zrow.clear();
    VB.cint4w_cn.clear();
    VB.c2d_cn.clear();
    VB.c3d_cn.clear();
    VB.last_int4_cn.clear();
    VB.last_bf16_cn.clear();
    VB.last_f32_cn.clear();
    for (int c = 0; c < 3; c++) m.qa0[c] = m.qa1[c] = CountNode();
    VB.bf16c.clear();
    VB.bf16c_hi.clear();
    std::fill(VB.bf_hi_prev.begin(), VB.bf_hi_prev.end(), CountNode());
    if (np2) {  // NP2: the groups move f32 rows, the f32 models start fresh too
      m.f32hl.assign(257 * 256, 1024);
      VB.f32hl_ci.clear();
      VB.f32hc.clear();
      std::fill(VB.f32_hi_prev.begin(), VB.f32_hi_prev.end(), CountNode());
    }
  }
  BinEncoder EB;
  for (int i : ib) {
    std::string cn = HB.code(EB, es[i], enc[i]);
    encode_values(EB, VB, es[i], enc[i], cn);
  }
  std::vector<uint8_t> B = EB.finish();
  size_t K = std::min(capacity_bytes(gs), B.size() - 1);
  if (B[0] != 0) die("stream B does not start with the zero cache byte");
  // the carried number; the reader folds the digits in file order (Horner
  // x = x * radix + digit), so they come out of X from the last one
  Big X;
  for (size_t j = K; j-- > 0;) X.mul_add(256, B[1 + j]);
  std::vector<std::vector<uint32_t>> digs(gs.size());
  for (size_t g = 0; g < gs.size(); g++) digs[g].resize(gs[g].n);
  for (size_t g = gs.size(); g-- > 0;)
    for (int j = gs[g].n; j-- > 0;) digs[g][j] = X.divmod(uint32_t(gs[g].n - j));
  if (!X.zero()) die("carried number does not fit");
  for (size_t gi = 0; gi < gs.size(); gi++) {
    const Group& G = gs[gi];
    const size_t kb = key_bytes(es, G);
    std::vector<int> idx(G.n);
    for (int j = 0; j < G.n; j++) idx[j] = j;
    std::sort(idx.begin(), idx.end(), [&](int x, int y) {
      return std::memcmp(key_row(es, G, x), key_row(es, G, y), kb) < 0;
    });
    for (int j = 1; j < G.n; j++)
      if (!std::memcmp(key_row(es, G, idx[j - 1]), key_row(es, G, idx[j]), kb))
        die("group %s: two equal key rows, the order is not readable", G.name.c_str());
    std::vector<int> perm(G.n);
    for (int j = 0; j < G.n; j++) {
      uint32_t d = digs[gi][j];
      perm[j] = idx[d];
      idx.erase(idx.begin() + d);
    }
    apply_perm(es, G, perm);
  }
  // stream A, with the permuted tensors
  BinEncoder EA;
  HeadCoder H;
  ValModels VA;
  for (int i : ia) {
    std::string cn = H.code(EA, es[i], uint8_t(enc[i] | keyflag[i]));
    encode_values(EA, VA, es[i], enc[i], cn, keyflag[i]);
  }
  std::vector<uint8_t> A = EA.finish();
  if (A[0] != 0) die("stream A does not start with the zero cache byte");
  std::vector<uint8_t> out(8);
  std::memcpy(out.data(), np2 ? (ver >= 6 ? "FX2TFWP6" : "FX2TFWP5") : (ver >= 6 ? "FX2TFWN6" : "FX2TFWN5"), 8);
  auto put = [&](uint32_t v) {
    auto b = u32le(v);
    out.insert(out.end(), b.begin(), b.end());
  };
  put(uint32_t(es.size()));
  put(uint32_t(ia.size()));
  put(uint32_t(A.size() - 1));
  put(uint32_t(K));
  out.insert(out.end(), A.begin() + 1, A.end());
  out.insert(out.end(), B.begin() + 1 + K, B.end());
  *K_out = K;
  *permuted_out = es;
  return out;
}

}  // namespace
}  // namespace fx2

using namespace fx2;

static std::vector<uint8_t> slurp(const char* p) {
  FILE* f = std::fopen(p, "rb");
  if (!f) die("cannot open %s", p);
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(size_t(sz > 0 ? sz : 0));
  if (std::fread(b.data(), 1, b.size(), f) != b.size()) die("short read %s", p);
  std::fclose(f);
  return b;
}

static void spit(const char* p, const std::vector<uint8_t>& b) {
  FILE* f = std::fopen(p, "wb");
  if (!f) die("cannot write %s", p);
  if (std::fwrite(b.data(), 1, b.size(), f) != b.size()) die("short write %s", p);
  std::fclose(f);
}

// identity probe: version and ENC_F32HC of the input, from the plain writer
static void identity(const char* in, const std::vector<Ent>& es, int* ver, bool* k) {
  std::vector<uint8_t> orig = slurp(in);
  if (orig.size() < 8) die("%s: too small", in);
  if (!std::memcmp(orig.data(), "FX2TFWC5", 8)) *ver = 5;
  else if (!std::memcmp(orig.data(), "FX2TFWC6", 8)) *ver = 6;
  else die("%s: not a v5 or v6 container (magic %.8s)", in, reinterpret_cast<const char*>(orig.data()));
  for (int kk = 0; kk < 2; kk++)
    if (encode_plain(es, *ver, kk == 1) == orig) {
      *k = kk == 1;
      std::printf("identity probe (v%d writer%s): BIT IDENTICAL, %zu B\n", *ver,
                  *k ? ", ENC_F32HC" : "", orig.size());
      return;
    }
  die("STOP: the v%d writer of this tool does NOT reproduce %s bit for bit "
      "(with and without ENC_F32HC); nothing was written", *ver, in);
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "dump" && argc == 4) {
    std::vector<Ent> es = load_ents(argv[2]);
    FILE* f = std::fopen(argv[3], "wb");
    if (!f) die("cannot write %s", argv[3]);
    std::fwrite("FX2TFW01", 1, 8, f);
    uint32_t n = uint32_t(es.size());
    std::fwrite(&n, 4, 1, f);
    for (const auto& e : es) {
      uint32_t nl = uint32_t(e.name.size()), nd = uint32_t(e.shape.size());
      std::fwrite(&nl, 4, 1, f);
      std::fwrite(e.name.data(), 1, nl, f);
      std::fwrite(&e.dt, 1, 1, f);
      std::fwrite(&nd, 4, 1, f);
      std::fwrite(e.shape.data(), 4, nd, f);
      std::fwrite(e.data.data(), 1, e.data.size(), f);
    }
    std::fclose(f);
    std::printf("%u tensors\n", n);
    return 0;
  }
  if (mode == "ident" && argc == 3) {
    std::vector<Ent> es = load_ents(argv[2]);
    int ver;
    bool k;
    identity(argv[2], es, &ver, &k);
    return 0;
  }
  if (mode == "np" && argc >= 4) {
    const char* in = argv[2];
    const char* out = argv[3];
    int kopt = -1;  // -1: as the input
    bool fw = false, np2 = false, dead = false, mark = false;
    for (int a = 4; a < argc; a++) {
      std::string o = argv[a];
      if (o == "k") kopt = 1;
      else if (o == "nok") kopt = 0;
      else if (o == "fw") fw = true;
      else if (o == "np2") np2 = true;
      else if (o == "dead") dead = true;
      else if (o == "deadmark") dead = mark = true;  // markers also without np2 (engine probe)
      else die("unknown option %s", o.c_str());
    }
    FILE* ex = std::fopen(out, "rb");
    if (ex) {
      std::fclose(ex);
      die("refusing to overwrite %s", out);
    }
    std::vector<Ent> es = load_ents(in);
    int ver;
    bool kin;
    identity(in, es, &ver, &kin);
    bool k = kopt < 0 ? kin : kopt == 1;
    std::vector<uint8_t> plain_in = encode_plain(es, ver, k);
    int ndead = 0, qmark = 0;
    if (dead) {
      ndead = zero_dead_key_dims(es, np2 || mark, &qmark);
      std::printf("dead KDA key dims: %d zeroed (k conv, dt_bias, q conv; %d q conv marker rows)\n", ndead,
                  qmark);
    }
    std::vector<Group> gs = np_groups(es, fw, np2);
    size_t K;
    std::vector<Ent> perm;
    std::vector<uint8_t> np = encode_np(es, ver, k, gs, np2, &K, &perm);
    std::vector<uint8_t> plain = encode_plain(es, ver, k);
    spit(out, np);
    // read back with the reader of this tree
    std::vector<Ent> back = load_ents(out);
    bool same = back.size() == perm.size();
    for (size_t i = 0; same && i < back.size(); i++)
      same = back[i].name == perm[i].name && back[i].dt == perm[i].dt &&
             back[i].shape == perm[i].shape && back[i].data == perm[i].data;
    bool moved = false;
    for (size_t i = 0; i < es.size(); i++) moved |= es[i].data != perm[i].data;
    std::printf("read back %s against the permuted tensors: %s (%zu tensors), permutation %s\n",
                np2 ? "NP2" : "NP", same ? "BIT IDENTICAL" : "DIFFERENT", back.size(),
                moved ? "non-trivial" : "TRIVIAL");
    if (!same || !moved) {
      std::remove(out);
      die("STOP: output removed");
    }
    size_t so = slurp(in).size();
    std::printf("groups %zu%s%s, carried K = %zu B, plain v%d%s %zu B%s, %s %zu B (%+ld B against "
                "plain, form cost + K - plain %+ld B)\n",
                gs.size(), fw ? " (with the fast-weight layers)" : "", np2 ? " (with the KDA dims)" : "",
                K, ver, k ? "k" : "", plain.size(), dead ? " (after dead)" : "", np2 ? "NP2" : "NP",
                np.size(), long(np.size()) - long(plain.size()), long(np.size() + K) - long(plain.size()));
    if (dead)
      std::printf("dead: plain v%d%s %zu B before, %zu B after (%+ld B)\n", ver, k ? "k" : "",
                  plain_in.size(), plain.size(), long(plain.size()) - long(plain_in.size()));
    std::printf("output %s: %zu B, %+ld B against the input (%+ld B in S, the container "
                "ships twice)\n",
                out, np.size(), long(np.size()) - long(so), 2 * (long(np.size()) - long(so)));
    std::printf("the compressor must be built with NEIF_TF1_WC5=1%s%s to read this container.\n",
                k ? " and NEIF_TF1_WC5K=1" : "", np2 ? " and NEIF_TF1_NP2=1" : "");
    if (fw || np2)
      std::printf("WARNING %s: the engine results follow the stored order (%s); no bit-identical "
                  "archive against the input.\n", fw && np2 ? "fw np2" : (fw ? "fw" : "np2"),
                  fw ? "the fast-weight head reads the unit index" : "KDA sums at rounding level");
    return 0;
  }
  if (mode == "plain" && (argc == 4 || argc == 5)) {
    // the decoded tensors of any readable container as a plain v5/v6 one
    // (for readers without format NP); default ENC_F32HC, nok: ENC_F32HI
    FILE* ex = std::fopen(argv[3], "rb");
    if (ex) {
      std::fclose(ex);
      die("refusing to overwrite %s", argv[3]);
    }
    std::vector<uint8_t> head = slurp(argv[2]);
    if (head.size() < 8) die("%s: too small", argv[2]);
    const int ver = head[7] == '6' ? 6 : 5;
    std::vector<Ent> es = load_ents(argv[2]);
    bool k = true;
    if (argc == 5) {
      if (!std::strcmp(argv[4], "nok")) k = false;
      else if (std::strcmp(argv[4], "k")) die("unknown option %s", argv[4]);
    }
    std::vector<uint8_t> p = encode_plain(es, ver, k);
    spit(argv[3], p);
    std::vector<Ent> back = load_ents(argv[3]);
    bool same = back.size() == es.size();
    for (size_t i = 0; same && i < back.size(); i++)
      same = back[i].name == es[i].name && back[i].data == es[i].data;
    if (!same) {
      std::remove(argv[3]);
      die("STOP: plain read back differs, output removed");
    }
    std::printf("plain v%d%s %s: %zu B, read back BIT IDENTICAL (%zu tensors)\n", ver, k ? "k" : "",
                argv[3], p.size(), es.size());
    return 0;
  }
  if (mode == "qat-snap" && (argc >= 4 && argc <= 28)) {
    const char* in = argv[2];
    const char* out_path = argv[3];
    int min_blk = (argc >= 5) ? std::atoi(argv[4]) : 0;
    int up_min_blk = (argc >= 6) ? std::atoi(argv[5]) : 0;
    int up_max_blk = (argc >= 7) ? std::atoi(argv[6]) : 16;
    std::vector<uint8_t> raw_in = slurp(in);
    if (raw_in.size() < 24) die("%s: too small", in);
    uint32_t orig_K = 0;
    if (!std::memcmp(raw_in.data(), "FX2TFW01", 8)) orig_K = 20277;
    else std::memcpy(&orig_K, &raw_in[20], 4);
    std::vector<Ent> es = load_ents(in);
    std::vector<Group> gs = np_groups(es, true, true);
    std::vector<uint8_t> keyflag(es.size(), 0);
    size_t last = 0;
    std::vector<std::vector<size_t>> prot(es.size());
    for (size_t i = 0; i < es.size(); i++) prot[i].assign(es[i].rows(), 0);
    for (const auto& G : gs) {
      for (auto p : G.rows_at) last = std::max(last, size_t(p.first));
      for (auto p : G.elem_at) last = std::max(last, size_t(p.first));
      for (auto p : G.cols_at) last = std::max(last, size_t(p.first));
      last = std::max(last, size_t(G.key));
      keyflag[G.key] = G.flag;
      const size_t kb = key_bytes(es, G);
      std::vector<int> idx(G.n);
      for (int j = 0; j < G.n; j++) idx[j] = j;
      std::sort(idx.begin(), idx.end(), [&](int x, int y) {
        return std::memcmp(key_row(es, G, x), key_row(es, G, y), kb) < 0;
      });
      for (int m = 0; m < G.n; m++) {
        size_t p_need = 0;
        if (m > 0) {
          const uint8_t* a = key_row(es, G, idx[m - 1]);
          const uint8_t* b = key_row(es, G, idx[m]);
          size_t p = 0;
          while (p < kb && a[p] == b[p]) p++;
          p_need = std::max(p_need, p + 1);
        }
        if (m + 1 < G.n) {
          const uint8_t* a = key_row(es, G, idx[m]);
          const uint8_t* b = key_row(es, G, idx[m + 1]);
          size_t p = 0;
          while (p < kb && a[p] == b[p]) p++;
          p_need = std::max(p_need, p + 1);
        }
        prot[G.key][G.r0 + idx[m]] = p_need;
      }
    }
    auto bf16_val = [](uint16_t b) -> float {
      uint32_t u = uint32_t(b) << 16;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    };
    const double kPi = std::acos(-1.0);
    const double lam_up = (argc >= 11) ? std::atof(argv[10]) : 2.18;
    const double lam_dn = (argc >= 12) ? std::atof(argv[11]) : 1.01;
    const double alpha_up = (argc >= 13) ? std::atof(argv[12]) : 1.58;
    const double alpha_dn = (argc >= 14) ? std::atof(argv[13]) : 1.19;
    const double lam_qk = (argc >= 15) ? std::atof(argv[14]) : 0.45;
    const double alpha_qk = (argc >= 16) ? std::atof(argv[15]) : 1.10;
    const double lam_v = (argc >= 17) ? std::atof(argv[16]) : 0.60;
    const double alpha_v = (argc >= 18) ? std::atof(argv[17]) : 1.12;
    const double lam_o = (argc >= 19) ? std::atof(argv[18]) : 0.20;
    const double alpha_o = (argc >= 20) ? std::atof(argv[19]) : 1.00;
    const double lam_g = (argc >= 21) ? std::atof(argv[20]) : 0.95;
    const double alpha_g = (argc >= 22) ? std::atof(argv[21]) : 1.19;
    const double lam_emb = (argc >= 23) ? std::atof(argv[22]) : 0.35;
    const double alpha_emb = (argc >= 24) ? std::atof(argv[23]) : 1.06;
    const double w_sq_up_k = 3.0, w_sq_g_k = 8.0, w_sq_qk_k = 8.0, w_sq_v_k = 8.0, ridge_o_k = 0.25;
    const int min_lv_35 = 0;
    auto f32_to_bf16_4bit = [](float f) -> uint16_t {
      uint32_t u;
      std::memcpy(&u, &f, 4);
      uint16_t b = uint16_t((u + 0x8000u) >> 16);
      return uint16_t((uint32_t(b) + 8u) & 0xFFF0u);
    };
    auto r_bits = [](int v) -> double {
      int a = std::abs(v);
      return (a == 0) ? 0.0 : ((a == 2) ? 2.1 : ((a == 4) ? 3.4 : 4.0));
    };
    // Pre-snap mantissa quantization (4 bits masked) for Stream A ENC_BF16C row scales
    // and non-key ENC_F32HC tensors, plus L2/norm/dot/sum-preserving grid snapping for
    // embedding, value_embeddings, and prior_embedding weight tensors.
    for (size_t i = 0; i <= last; i++) {
      Ent& e = es[i];
      if (e.dt == DT_BF16 && ends_with(e.name, ".weight.scale")) {
        uint16_t* u = reinterpret_cast<uint16_t*>(e.data.data());
        for (size_t k = 0; k < e.numel(); k++) u[k] = uint16_t((uint32_t(u[k]) + 8u) & 0xFFF0u);
      } else if (e.dt == DT_F32 && e.name.find("value_embedding_coefficients") == std::string::npos) {
        if (e.name.find("query_convolution.weight") != std::string::npos) {
          uint32_t* u = reinterpret_cast<uint32_t*>(e.data.data());
          const size_t kb = 16;
          for (int h = 0; h < 3; h++) {
            std::vector<uint32_t> qc_digs(64, 0);
            for (size_t r1 = 0; r1 < 64; r1++) {
              uint32_t c = 0;
              for (size_t r2 = r1 + 1; r2 < 64; r2++)
                c += std::memcmp(&e.data[(64 * h + r2) * kb], &e.data[(64 * h + r1) * kb], kb) < 0;
              qc_digs[r1] = c;
            }
            for (size_t r = 64 * h; r < 64 * (h + 1); r++) {
              for (size_t col = 0; col < 4; col++) {
                uint32_t hi16 = u[r * 4 + col] >> 16;
                if (hi16 > 0 && hi16 < 256) u[r * 4 + col] = (hi16 << 4) << 16;
                else if (hi16 >= 256) u[r * 4 + col] = ((hi16 + 8u) & 0xFFF0u) << 16;
              }
            }
            std::vector<int> idx(64);
            for (int j = 0; j < 64; j++) idx[j] = j;
            std::sort(idx.begin(), idx.end(), [&](int x, int y) {
              return std::memcmp(&e.data[(64 * h + x) * kb], &e.data[(64 * h + y) * kb], kb) < 0;
            });
            std::vector<int> perm(64);
            for (int j = 0; j < 64; j++) {
              uint32_t d = qc_digs[j];
              perm[j] = idx[d];
              idx.erase(idx.begin() + d);
            }
            for (const auto& G : gs) {
              if (G.key == int(i) && G.r0 == 64 * h) {
                apply_perm(es, G, perm);
                break;
              }
            }
          }
        } else {
          uint32_t* u = reinterpret_cast<uint32_t*>(e.data.data());
          for (size_t k = 0; k < e.numel(); k++) {
            uint32_t hi16 = u[k] >> 16;
            if (hi16) u[k] = ((hi16 + 8u) & 0xFFF0u) << 16;
          }
        }
      } else if (e.dt == DT_I8 && (e.name.compare(0, 10, "embedding.") == 0 ||
                                   e.name.compare(0, 17, "value_embeddings.") == 0 ||
                                   e.name.compare(0, 16, "prior_embedding.") == 0)) {
        size_t R = e.rows(), C = e.cols();
        for (size_t r = 0; r < R; r++) {
          int8_t* row = reinterpret_cast<int8_t*>(&e.data[r * C]);
          std::vector<int> orig_u(C);
          double sq_orig = 0.0;
          for (size_t c = 0; c < C; c++) { orig_u[c] = row[c]; sq_orig += double(row[c]) * row[c]; }
          if (sq_orig <= 0.0) continue;
          double w_sq = 8.0 / sq_orig, w_dot = 8.0 / sq_orig, w_sum = 2.0 / double(C);
          double dsq = 0.0, ddot = 0.0, dsum = 0.0;
          auto eval_cost = [&](size_t c, int v_cur, int v_new) {
            double dv = v_new - v_cur;
            double e_cur = v_cur - orig_u[c], e_new = v_new - orig_u[c];
            double d_sq_new = dsq + (double(v_new) * v_new - double(v_cur) * v_cur);
            double d_dot_new = ddot + dv * orig_u[c];
            double d_sum_new = dsum + dv;
            return (e_new * e_new - e_cur * e_cur)
                 + w_sq * (d_sq_new * d_sq_new - dsq * dsq)
                 + w_dot * (d_dot_new * d_dot_new - ddot * ddot)
                 + w_sum * (d_sum_new * d_sum_new - dsum * dsum);
          };
          auto snap_u = [&](size_t c, int vn_lo, int vn_hi) {
            int v = row[c];
            int vn = (eval_cost(c, v, vn_hi) <= eval_cost(c, v, vn_lo)) ? vn_hi : vn_lo;
            double dv = vn - v;
            dsq += double(vn) * vn - double(v) * v;
            ddot += dv * orig_u[c];
            dsum += dv;
            row[c] = int8_t(vn);
          };
          std::vector<size_t> c1, c35;
          for (size_t c = 0; c < C; c++) {
            int a = std::abs(int(row[c]));
            if (a == 1) c1.push_back(c);
            else if (a == 3 || a == 5 || a == 6) c35.push_back(c);
          }
          for (size_t c : c1) snap_u(c, 0, 2 * ((row[c] > 0) ? 1 : -1));
          for (size_t c : c35) {
            int s_u = (row[c] > 0) ? 1 : -1, a = std::abs(int(row[c]));
            int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
            snap_u(c, s_u * lo, s_u * hi);
          }
          for (int pass = 0; pass < 4; pass++) {
            for (size_t c : c1) {
              int s_u = (orig_u[c] > 0) ? 1 : -1, v_cur = row[c], v_alt = (v_cur == 0) ? (2 * s_u) : 0;
              if (eval_cost(c, v_cur, v_alt) < -1e-12) {
                double d_alt = v_alt - v_cur;
                dsq += double(v_alt) * v_alt - double(v_cur) * v_cur;
                ddot += d_alt * orig_u[c];
                dsum += d_alt;
                row[c] = int8_t(v_alt);
              }
            }
            for (size_t c : c35) {
              int s_u = (orig_u[c] > 0) ? 1 : -1, a = std::abs(orig_u[c]), v_cur = row[c];
              int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
              int v_alt = (v_cur == s_u * lo) ? (s_u * hi) : (s_u * lo);
              if (eval_cost(c, v_cur, v_alt) < -1e-12) {
                double d_alt = v_alt - v_cur;
                dsq += double(v_alt) * v_alt - double(v_cur) * v_cur;
                ddot += d_alt * orig_u[c];
                dsum += d_alt;
                row[c] = int8_t(v_alt);
              }
            }
          }
        }
      }
    }
    #pragma omp parallel for schedule(dynamic)
    for (int blk = min_blk; blk < 16; blk++) {
      std::string up_q_name = "blocks." + std::to_string(blk) + ".mlp.up.weight.q";
      std::string up_s_name = "blocks." + std::to_string(blk) + ".mlp.up.weight.scale";
      std::string dn_q_name = "blocks." + std::to_string(blk) + ".mlp.down.weight.q";
      int i_up_q = -1, i_up_s = -1, i_dn_q = -1;
      for (size_t i = 0; i <= last; i++) {
        if (es[i].name == up_q_name) i_up_q = (int)i;
        if (es[i].name == up_s_name) i_up_s = (int)i;
        if (es[i].name == dn_q_name) i_dn_q = (int)i;
      }
      if (i_up_q < 0 || i_up_s < 0 || i_dn_q < 0) continue;
      const size_t N_neur = 1024, D = 192;
      const uint16_t* up_sc = reinterpret_cast<const uint16_t*>(es[i_up_s].data.data());
      std::vector<double> a_orig(N_neur, 0.0);
      std::vector<double> u_orig(N_neur * D, 0.0);
      for (size_t j = 0; j < N_neur; j++) {
        const int8_t* urow = reinterpret_cast<const int8_t*>(&es[i_up_q].data[j * D]);
        double sq = 0.0;
        for (size_t d = 0; d < D; d++) sq += double(urow[d]) * urow[d];
        double sc = bf16_val(up_sc[j]); a_orig[j] = sq * sc * sc;
        double inv = (sq > 0.0) ? (1.0 / std::sqrt(sq)) : 0.0;
        for (size_t d = 0; d < D; d++) u_orig[j * D + d] = urow[d] * inv;
      }

      if (blk >= up_min_blk && blk < up_max_blk) {
        std::vector<double> col_imp(N_neur, 0.0);
        for (size_t r = 0; r < D; r++) {
          const int8_t* drow = reinterpret_cast<const int8_t*>(&es[i_dn_q].data[r * N_neur]);
          for (size_t j = 0; j < N_neur; j++) col_imp[j] += double(drow[j]) * drow[j];
        }
        std::vector<double> G_up(D * D, 0.0);
        for (size_t j = 0; j < N_neur; j++) {
          double wj = a_orig[j] * std::sqrt(col_imp[j] + 1.0);
          const double* uj = &u_orig[j * D];
          for (size_t d1 = 0; d1 < D; d1++) {
            for (size_t d2 = d1; d2 < D; d2++) {
              G_up[d1 * D + d2] += wj * uj[d1] * uj[d2];
            }
          }
        }
        double trG = 0.0;
        for (size_t d = 0; d < D; d++) trG += G_up[d * D + d];
        double inv_trG = (trG > 0.0) ? (double(D) / trG) : 1.0;
        for (size_t d1 = 0; d1 < D; d1++) {
          for (size_t d2 = d1; d2 < D; d2++) {
            double v = G_up[d1 * D + d2] * inv_trG + ((d1 == d2) ? 1.5 : 0.0);
            G_up[d1 * D + d2] = G_up[d2 * D + d1] = v;
          }
        }
        std::vector<uint32_t> mlp_digs(N_neur, 0);
        for (size_t k = 0; k < N_neur; k++) {
          uint32_t c = 0;
          for (size_t j = k + 1; j < N_neur; j++)
            c += std::memcmp(&es[i_up_q].data[j * D], &es[i_up_q].data[k * D], D) < 0;
          mlp_digs[k] = c;
        }
        std::vector<int> up_idx(N_neur), up_pos(N_neur);
        for (size_t j = 0; j < N_neur; j++) up_idx[j] = int(j);
        std::sort(up_idx.begin(), up_idx.end(), [&](int x, int y) {
          return std::memcmp(&es[i_up_q].data[size_t(x) * D], &es[i_up_q].data[size_t(y) * D], D) < 0;
        });
        for (size_t m = 0; m < N_neur; m++) up_pos[up_idx[m]] = int(m);
        auto order_ok = [&](size_t j) {
          const uint8_t* cur = &es[i_up_q].data[j * D];
          for (size_t o = 0; o < N_neur; o++)
            if (o != j && std::memcmp(&es[i_up_q].data[o * D], cur, D) == 0) return false;
          return true;
        };
        uint16_t* up_sc_mut = reinterpret_cast<uint16_t*>(es[i_up_s].data.data());
        for (size_t j = 0; j < N_neur; j++) {
          int8_t* urow = reinterpret_cast<int8_t*>(&es[i_up_q].data[j * D]);
          float sc_orig = bf16_val(up_sc_mut[j]);
          uint16_t b_new = (alpha_up > 1.0) ? f32_to_bf16_4bit(float(sc_orig * alpha_up)) : up_sc_mut[j];
          float sc_new = bf16_val(b_new);
          double inv_alpha = (sc_new > 0.0f && sc_orig > 0.0f) ? (double(sc_orig) / double(sc_new)) : 1.0;
          up_sc_mut[j] = b_new;
          std::vector<double> orig_u(D);
          std::vector<int> raw_u(D);
          double sq_orig = 0.0;
          for (size_t d = 0; d < D; d++) {
            raw_u[d] = urow[d];
            orig_u[d] = double(urow[d]) * inv_alpha;
            sq_orig += orig_u[d] * orig_u[d];
          }
          if (sq_orig <= 0.0) continue;
          double w_sq = 3.0 / sq_orig, w_dot = 3.0 / sq_orig, w_sum = 1.0 / double(D);
          std::vector<double> g(D, 0.0);
          double dsq = 0.0, ddot = 0.0, dsum = 0.0;
          for (size_t c = 0; c < D; c++) {
            double err0 = double(urow[c]) - orig_u[c];
            dsq += double(urow[c]) * urow[c] - orig_u[c] * orig_u[c];
            ddot += err0 * orig_u[c];
            dsum += err0;
            if (err0 != 0.0) {
              const double* mc = &G_up[c * D];
              for (size_t d = 0; d < D; d++) g[d] += err0 * mc[d];
            }
          }
          auto eval_cost = [&](size_t c, int v_cur, int v_new) {
            double dv = v_new - v_cur;
            double d_sq_new = dsq + (double(v_new) * v_new - double(v_cur) * v_cur);
            double d_dot_new = ddot + dv * orig_u[c];
            double d_sum_new = dsum + dv;
            return 2.0 * dv * g[c] + dv * dv * G_up[c * D + c]
                 + w_sq * (d_sq_new * d_sq_new - dsq * dsq)
                 + w_dot * (d_dot_new * d_dot_new - ddot * ddot)
                 + w_sum * (d_sum_new * d_sum_new - dsum * dsum)
                 + lam_up * (r_bits(v_new) - r_bits(v_cur));
          };
          auto snap_u = [&](size_t c, int vn_lo, int vn_hi) {
            int v = urow[c];
            int vn1 = (eval_cost(c, v, vn_hi) <= eval_cost(c, v, vn_lo)) ? vn_hi : vn_lo;
            int vn2 = (vn1 == vn_hi) ? vn_lo : vn_hi;
            int vn = v;
            urow[c] = int8_t(vn1);
            if (order_ok(j)) {
              vn = vn1;
            } else {
              urow[c] = int8_t(vn2);
              if (order_ok(j)) vn = vn2;
              else urow[c] = int8_t(v);
            }
            double dv = vn - v;
            dsq += double(vn) * vn - double(v) * v;
            ddot += dv * orig_u[c];
            dsum += dv;
            urow[c] = int8_t(vn);
            const double* mc = &G_up[c * D];
            for (size_t d = 0; d < D; d++) g[d] += dv * mc[d];
          };
          std::vector<size_t> c1, c35, c247;
          for (size_t c = 0; c < D; c++) {
            int a = std::abs(raw_u[c]);
            if (a == 1) c1.push_back(c);
            else if (a == 3 || a == 5 || a == 6) c35.push_back(c);
            else if (a == 2 || a == 4 || a == 7) c247.push_back(c);
          }
          size_t p_j = prot[i_up_q][j];
          auto cmp_u = [&](size_t a, size_t b) {
            bool ap = (a < p_j), bp = (b < p_j);
            if (ap != bp) return !ap;
            return G_up[a * D + a] < G_up[b * D + b];
          };
          std::sort(c1.begin(), c1.end(), cmp_u);
          std::sort(c35.begin(), c35.end(), cmp_u);
          std::sort(c247.begin(), c247.end(), cmp_u);
          for (size_t c : c1) snap_u(c, 0, 2 * ((raw_u[c] > 0) ? 1 : -1));
          for (size_t c : c35) {
            int s_u = (raw_u[c] > 0) ? 1 : -1, a = std::abs(raw_u[c]);
            int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
            snap_u(c, s_u * lo, s_u * hi);
          }
          auto try_multi_u = [&](size_t c) {
            int a = std::abs(raw_u[c]);
            if (a == 0) return;
            int s_u = (raw_u[c] > 0) ? 1 : -1, v_cur = urow[c];
            int hi = (a <= 2) ? 2 : ((a <= 4) ? 4 : 7);
            const int levels[4] = {0, 2 * s_u, 4 * s_u, 7 * s_u};
            int best_v = v_cur;
            double best_cost = -1e-12;
            for (int lv : levels) {
              if (std::abs(lv) > hi || lv == v_cur) continue;
              double c_lv = eval_cost(c, v_cur, lv);
              if (c_lv < best_cost) {
                urow[c] = int8_t(lv);
                if (order_ok(j)) { best_cost = c_lv; best_v = lv; }
                urow[c] = int8_t(v_cur);
              }
            }
            if (best_v != v_cur) {
              double d_alt = best_v - v_cur;
              dsq += double(best_v) * best_v - double(v_cur) * v_cur;
              ddot += d_alt * orig_u[c];
              dsum += d_alt;
              urow[c] = int8_t(best_v);
              const double* mc = &G_up[c * D];
              for (size_t d = 0; d < D; d++) g[d] += d_alt * mc[d];
            }
          };
          for (int pass = 0; pass < 4; pass++) {
            for (size_t c : c1) try_multi_u(c);
            for (size_t c : c35) try_multi_u(c);
            for (size_t c : c247) try_multi_u(c);
          }
        }
        if (blk < 16) {
          std::vector<int> idx(N_neur);
          for (size_t j = 0; j < N_neur; j++) idx[j] = int(j);
          std::sort(idx.begin(), idx.end(), [&](int x, int y) {
            return std::memcmp(&es[i_up_q].data[size_t(x) * D], &es[i_up_q].data[size_t(y) * D], D) < 0;
          });
          std::vector<int> perm(N_neur);
          for (size_t j = 0; j < N_neur; j++) {
            uint32_t d = mlp_digs[j];
            perm[j] = idx[d];
            idx.erase(idx.begin() + d);
          }
          for (const auto& G : gs) {
            if (G.key == i_up_q) {
              apply_perm(es, G, perm);
              break;
            }
          }
          std::vector<double> old_a = a_orig, old_u = u_orig;
          for (size_t j = 0; j < N_neur; j++) {
            a_orig[j] = old_a[perm[j]];
            std::memcpy(&u_orig[j * D], &old_u[size_t(perm[j]) * D], D * sizeof(double));
          }
          up_sc = reinterpret_cast<const uint16_t*>(es[i_up_s].data.data());
        }
      }

      std::vector<double> a_neur(N_neur, 0.0);
      std::vector<double> u_norm(N_neur * D, 0.0);
      for (size_t j = 0; j < N_neur; j++) {
        const int8_t* urow = reinterpret_cast<const int8_t*>(&es[i_up_q].data[j * D]);
        double sq = 0.0;
        for (size_t d = 0; d < D; d++) sq += double(urow[d]) * urow[d];
        double sc = bf16_val(up_sc[j]); a_neur[j] = sq * sc * sc;
        double inv = (sq > 0.0) ? (1.0 / std::sqrt(sq)) : 0.0;
        for (size_t d = 0; d < D; d++) u_norm[j * D + d] = urow[d] * inv;
      }
      std::vector<double> cov(N_neur * N_neur, 0.0);
      std::vector<double> cov_cross(N_neur * N_neur, 0.0);
      for (size_t i = 0; i < N_neur; i++) {
        const double* ui = &u_norm[i * D];
        cov[i * N_neur + i] = 0.375 * a_neur[i] * a_neur[i];
        for (size_t j = i + 1; j < N_neur; j++) {
          const double* uj = &u_norm[j * D];
          double rho = 0.0;
          for (size_t d = 0; d < D; d++) rho += ui[d] * uj[d];
          rho = std::max(-1.0, std::min(1.0, rho));
          double th = std::acos(rho);
          double f_rho = ((1.0 + 2.0 * rho * rho) * (kPi - th) + 3.0 * rho * std::sin(th)) / (8.0 * kPi);
          double c_ij = a_neur[i] * a_neur[j] * f_rho;
          cov[i * N_neur + j] = cov[j * N_neur + i] = c_ij;
        }
        if (blk >= up_min_blk && blk < up_max_blk) {
          for (size_t j = 0; j < N_neur; j++) {
            const double* uj_orig = &u_orig[j * D];
            double rho = 0.0;
            for (size_t d = 0; d < D; d++) rho += ui[d] * uj_orig[d];
            rho = std::max(-1.0, std::min(1.0, rho));
            double th = std::acos(rho);
            double f_rho = ((1.0 + 2.0 * rho * rho) * (kPi - th) + 3.0 * rho * std::sin(th)) / (8.0 * kPi);
            cov_cross[i * N_neur + j] = a_neur[i] * a_orig[j] * f_rho;
          }
        }
      }
      std::vector<size_t> neur_rank(N_neur), inv_rank(N_neur);
      for (size_t j = 0; j < N_neur; j++) neur_rank[j] = j;
      std::sort(neur_rank.begin(), neur_rank.end(), [&](size_t a, size_t b) { return a_neur[a] < a_neur[b]; });
      for (size_t r = 0; r < N_neur; r++) inv_rank[neur_rank[r]] = r;
      std::string dn_s_name = "blocks." + std::to_string(blk) + ".mlp.down.weight.scale";
      int i_dn_s = -1;
      for (size_t i = 0; i <= last; i++) if (es[i].name == dn_s_name) i_dn_s = int(i);
      uint16_t* dn_sc_mut = (i_dn_s >= 0) ? reinterpret_cast<uint16_t*>(es[i_dn_s].data.data()) : nullptr;
      for (size_t r = 0; r < D; r++) {
        int8_t* drow = reinterpret_cast<int8_t*>(&es[i_dn_q].data[r * N_neur]);
        double inv_alpha_dn = 1.0;
        if (dn_sc_mut && alpha_dn > 1.0) {
          float sc_orig = bf16_val(dn_sc_mut[r]);
          uint16_t b_new = f32_to_bf16_4bit(float(sc_orig * alpha_dn));
          float sc_new = bf16_val(b_new);
          if (sc_new > 0.0f && sc_orig > 0.0f) inv_alpha_dn = double(sc_orig) / double(sc_new);
          dn_sc_mut[r] = b_new;
        }
        std::vector<int> orig_w(N_neur);
        for (size_t j = 0; j < N_neur; j++) orig_w[j] = drow[j];
        size_t c0 = prot[i_dn_q][r];
        std::vector<double> g(N_neur, 0.0);
        if (blk >= up_min_blk && blk < up_max_blk) {
          for (size_t i = 0; i < N_neur; i++) {
            const double* ci = &cov[i * N_neur];
            const double* cci = &cov_cross[i * N_neur];
            double s_g = 0.0;
            for (size_t j = 0; j < N_neur; j++) s_g += (ci[j] - inv_alpha_dn * cci[j]) * orig_w[j];
            g[i] = s_g;
          }
        }
        double mean_cov = 0.0;
        for (size_t j = 0; j < N_neur; j++) mean_cov += cov[j * N_neur + j];
        mean_cov /= double(N_neur);
        auto snap_col = [&](size_t c, int vn_lo, int vn_hi) {
          int v = drow[c];
          double d_lo = vn_lo - v, d_hi = vn_hi - v, c_cc = cov[c * N_neur + c];
          double cost_hi = 2.0 * d_hi * g[c] + d_hi * d_hi * c_cc + lam_dn * mean_cov * r_bits(vn_hi);
          double cost_lo = 2.0 * d_lo * g[c] + d_lo * d_lo * c_cc + lam_dn * mean_cov * r_bits(vn_lo);
          int vn = (cost_hi <= cost_lo) ? vn_hi : vn_lo;
          double dv = vn - v;
          drow[c] = int8_t(vn);
          const double* cov_c = &cov[c * N_neur];
          for (size_t j = 0; j < N_neur; j++) g[j] += dv * cov_c[j];
        };
        std::vector<size_t> cands1, cands356, cands247;
        for (size_t c = c0; c < N_neur; c++) {
          int a = std::abs(int(drow[c]));
          if (a == 1) cands1.push_back(c);
          else if (a == 3 || a == 5 || a == 6) cands356.push_back(c);
          else if (a == 2 || a == 4 || a == 7) cands247.push_back(c);
        }
        std::sort(cands1.begin(), cands1.end(), [&](size_t a, size_t b) { return inv_rank[a] < inv_rank[b]; });
        std::sort(cands356.begin(), cands356.end(), [&](size_t a, size_t b) { return inv_rank[a] < inv_rank[b]; });
        std::sort(cands247.begin(), cands247.end(), [&](size_t a, size_t b) { return inv_rank[a] < inv_rank[b]; });
        for (size_t c : cands1) snap_col(c, 0, 2 * ((drow[c] > 0) ? 1 : -1));
        for (size_t c : cands356) {
          int s = (drow[c] > 0) ? 1 : -1, a = std::abs(int(drow[c]));
          int lo = (a == 3) ? 2 : 4;
          int hi = (a == 3) ? 4 : 7;
          snap_col(c, s * lo, s * hi);
        }
        auto try_multi_dn = [&](size_t c) {
          int a = std::abs(orig_w[c]);
          if (a == 0) return;
          int s = (orig_w[c] > 0) ? 1 : -1, v_cur = drow[c];
          int hi = (a <= 2) ? 2 : ((a <= 4) ? 4 : 7);
          const int levels[4] = {0, 2 * s, 4 * s, 7 * s};
          int best_v = v_cur;
          double best_cost = -1e-12;
          for (int lv : levels) {
            if (std::abs(lv) > hi || lv == v_cur) continue;
            double d_alt = lv - v_cur;
            double c_lv = 2.0 * d_alt * g[c] + d_alt * d_alt * cov[c * N_neur + c] + lam_dn * mean_cov * (r_bits(lv) - r_bits(v_cur));
            if (c_lv < best_cost) { best_cost = c_lv; best_v = lv; }
          }
          if (best_v != v_cur) {
            double d_alt = best_v - v_cur;
            drow[c] = int8_t(best_v);
            const double* cov_c = &cov[c * N_neur];
            for (size_t j = 0; j < N_neur; j++) g[j] += d_alt * cov_c[j];
          }
        };
        for (int pass = 0; pass < 4; pass++) {
          for (size_t c : cands1) try_multi_dn(c);
          for (size_t c : cands356) try_multi_dn(c);
          for (size_t c : cands247) try_multi_dn(c);
        }
      }
    }

    // Optional RDO on embedding/unembedding/value_embeddings/prior_embedding
    if (lam_emb > 0.0 || alpha_emb > 1.0) {
      for (size_t i = 0; i <= last; i++) {
        if (es[i].dt != DT_I8) continue;
        if (es[i].name.find("blocks.") == 0) continue;
        bool is_unemb = (es[i].name == "unembedding.weight.q");
        double cur_lam_emb = is_unemb ? (1.0 * lam_emb) : lam_emb;
        double cur_alpha_emb = is_unemb ? (1.0 + 1.0 * (alpha_emb - 1.0)) : alpha_emb;
        int i_sc = -1;
        std::string sc_name = es[i].name.substr(0, es[i].name.size() - 2) + ".scale";
        for (size_t k = 0; k <= last; k++) {
          if (es[k].name == sc_name) { i_sc = int(k); break; }
        }
        if (i_sc < 0) continue;
        const size_t R = es[i].rows(), C = es[i].cols();
        if (C < 16) continue;
        uint16_t* sc_mut = reinterpret_cast<uint16_t*>(es[i_sc].data.data());
        for (size_t r = 0; r < R; r++) {
          int8_t* row = reinterpret_cast<int8_t*>(&es[i].data[r * C]);
          int r_nnz = 0;
          for (size_t c = 0; c < C; c++) if (row[c]) r_nnz++;
          if (r_nnz == 0) continue;
          double inv_alpha = 1.0;
          if (cur_alpha_emb > 1.0 && es[i_sc].numel() == R) {
            float sc_orig = bf16_val(sc_mut[r]);
            uint16_t b_new = f32_to_bf16_4bit(float(sc_orig * cur_alpha_emb));
            float sc_new = bf16_val(b_new);
            if (sc_new > 0.0f && sc_orig > 0.0f) inv_alpha = double(sc_orig) / double(sc_new);
            sc_mut[r] = b_new;
          }
          std::vector<int> raw_u(C);
          std::vector<double> orig_u(C);
          double sq_orig = 0.0;
          for (size_t c = 0; c < C; c++) {
            raw_u[c] = row[c];
            orig_u[c] = double(row[c]) * inv_alpha;
            sq_orig += orig_u[c] * orig_u[c];
          }
          if (sq_orig <= 0.0) continue;
          double w_sq = 6.0 / sq_orig, w_dot = 6.0 / sq_orig, w_sum = 1.5 / double(C);
          double dsq = 0.0, ddot = 0.0, dsum = 0.0;
          for (size_t c = 0; c < C; c++) {
            double err0 = double(row[c]) - orig_u[c];
            dsq += double(row[c]) * row[c] - orig_u[c] * orig_u[c];
            ddot += err0 * orig_u[c];
            dsum += err0;
          }
          auto eval_cost = [&](size_t c, int v_cur, int v_new) {
            double dv = v_new - v_cur;
            double e_cur = v_cur - orig_u[c], e_new = v_new - orig_u[c];
            double d_sq_new = dsq + (double(v_new) * v_new - double(v_cur) * v_cur);
            double d_dot_new = ddot + dv * orig_u[c];
            double d_sum_new = dsum + dv;
            return (e_new * e_new - e_cur * e_cur)
                 + w_sq * (d_sq_new * d_sq_new - dsq * dsq)
                 + w_dot * (d_dot_new * d_dot_new - ddot * ddot)
                 + w_sum * (d_sum_new * d_sum_new - dsum * dsum)
                 + cur_lam_emb * (r_bits(v_new) - r_bits(v_cur));
          };
          std::vector<size_t> c1, c35, c247;
          for (size_t c = 0; c < C; c++) {
            int a = std::abs(raw_u[c]);
            if (a == 1) c1.push_back(c);
            else if (a == 3 || a == 5 || a == 6) c35.push_back(c);
            else if (a == 2 || a == 4 || a == 7) c247.push_back(c);
          }
          for (size_t c : c1) {
            int v = row[c], s_u = (raw_u[c] > 0) ? 1 : -1;
            int vn = (eval_cost(c, v, 2 * s_u) <= eval_cost(c, v, 0)) ? (2 * s_u) : 0;
            if (vn == 0 && v != 0 && r_nnz <= 1) vn = 2 * s_u;
            if (vn == 0 && v != 0) r_nnz--;
            double dv = vn - v;
            dsq += double(vn) * vn - double(v) * v;
            ddot += dv * orig_u[c];
            dsum += dv;
            row[c] = int8_t(vn);
          }
          for (size_t c : c35) {
            int v = row[c], s_u = (raw_u[c] > 0) ? 1 : -1, a = std::abs(raw_u[c]);
            int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
            int vn = (eval_cost(c, v, s_u * hi) <= eval_cost(c, v, s_u * lo)) ? (s_u * hi) : (s_u * lo);
            double dv = vn - v;
            dsq += double(vn) * vn - double(v) * v;
            ddot += dv * orig_u[c];
            dsum += dv;
            row[c] = int8_t(vn);
          }
          auto try_multi_emb = [&](size_t c) {
            int a = std::abs(raw_u[c]);
            if (a == 0) return;
            int s_u = (raw_u[c] > 0) ? 1 : -1, v_cur = row[c];
            int hi = (a <= 2) ? 2 : ((a <= 4) ? 4 : 7);
            int lo_min = (min_lv_35 && a >= 5) ? 2 : ((min_lv_35 == 2 && a >= 3) ? 2 : 0);
            const int levels[4] = {0, 2 * s_u, 4 * s_u, 7 * s_u};
            int best_v = v_cur;
            double best_cost = -1e-12;
            for (int lv : levels) {
              if (std::abs(lv) > hi || std::abs(lv) < lo_min || lv == v_cur) continue;
              if (lv == 0 && v_cur != 0 && r_nnz <= 1) continue;
              double c_lv = eval_cost(c, v_cur, lv);
              if (c_lv < best_cost) { best_cost = c_lv; best_v = lv; }
            }
            if (best_v != v_cur) {
              if (best_v == 0 && v_cur != 0) r_nnz--;
              else if (best_v != 0 && v_cur == 0) r_nnz++;
              double d_alt = best_v - v_cur;
              dsq += double(best_v) * best_v - double(v_cur) * v_cur;
              ddot += d_alt * orig_u[c];
              dsum += d_alt;
              row[c] = int8_t(best_v);
            }
          };
          for (int pass = 0; pass < 3; pass++) {
            for (size_t c : c1) try_multi_emb(c);
            for (size_t c : c35) try_multi_emb(c);
            for (size_t c : c247) try_multi_emb(c);
          }
        }
      }
    }

    // Coupled linear-bottleneck QAT snapping for KDA gate projections
    #pragma omp parallel for schedule(dynamic)
    for (int blk = min_blk; blk < 16; blk++) {
      if (blk % 4 == 3) continue; // window attention layer has no gates
      for (const char* gname : {"forget_gate_projection", "output_gate_projection"}) {
        std::string pref = "blocks." + std::to_string(blk) + ".attention." + gname;
        int i_up_q = -1, i_up_s = -1, i_dn_q = -1, i_dn_s = -1;
        for (size_t i = 0; i <= last; i++) {
          if (es[i].name == pref + ".up.weight.q") i_up_q = int(i);
          if (es[i].name == pref + ".up.weight.scale") i_up_s = int(i);
          if (es[i].name == pref + ".down.weight.q") i_dn_q = int(i);
          if (es[i].name == pref + ".down.weight.scale") i_dn_s = int(i);
        }
        if (i_up_q < 0 || i_up_s < 0 || i_dn_q < 0 || i_dn_s < 0) continue;
        const size_t N_bot = es[i_up_q].rows(), D = es[i_up_q].cols();
        uint16_t* up_sc_mut = reinterpret_cast<uint16_t*>(es[i_up_s].data.data());
        uint16_t* dn_sc_mut = reinterpret_cast<uint16_t*>(es[i_dn_s].data.data());
        std::vector<uint32_t> bot_digs(N_bot, 0);
        for (size_t k = 0; k < N_bot; k++) {
          uint32_t c = 0;
          for (size_t j = k + 1; j < N_bot; j++)
            c += std::memcmp(&es[i_up_q].data[j * D], &es[i_up_q].data[k * D], D) < 0;
          bot_digs[k] = c;
        }
        std::vector<double> u_orig(N_bot * D, 0.0);
        for (size_t j = 0; j < N_bot; j++) {
          double sc = bf16_val(up_sc_mut[j]);
          const int8_t* urow = reinterpret_cast<const int8_t*>(&es[i_up_q].data[j * D]);
          for (size_t d = 0; d < D; d++) u_orig[j * D + d] = sc * urow[d];
        }
        auto order_ok = [&](size_t j) {
          const uint8_t* cur = &es[i_up_q].data[j * D];
          for (size_t o = 0; o < N_bot; o++)
            if (o != j && std::memcmp(&es[i_up_q].data[o * D], cur, D) == 0) return false;
          return true;
        };
        for (size_t j = 0; j < N_bot; j++) {
          int8_t* urow = reinterpret_cast<int8_t*>(&es[i_up_q].data[j * D]);
          float sc_orig = bf16_val(up_sc_mut[j]);
          uint16_t b_new = (alpha_g > 1.0) ? f32_to_bf16_4bit(float(sc_orig * alpha_g)) : up_sc_mut[j];
          float sc_new = bf16_val(b_new);
          double inv_alpha = (sc_new > 0.0f && sc_orig > 0.0f) ? (double(sc_orig) / double(sc_new)) : 1.0;
          up_sc_mut[j] = b_new;
          std::vector<int> raw_u(D);
          std::vector<double> orig_u(D);
          double sq_orig = 0.0;
          int r_nnz = 0;
          for (size_t d = 0; d < D; d++) {
            raw_u[d] = urow[d];
            if (urow[d]) r_nnz++;
            orig_u[d] = double(urow[d]) * inv_alpha;
            sq_orig += orig_u[d] * orig_u[d];
          }
          if (sq_orig <= 0.0) continue;
          double w_sq = w_sq_g_k / sq_orig, w_dot = w_sq_g_k / sq_orig, w_sum = 2.0 / double(D);
          double dsq = 0.0, ddot = 0.0, dsum = 0.0;
          for (size_t c = 0; c < D; c++) {
            double err0 = double(urow[c]) - orig_u[c];
            dsq += double(urow[c]) * urow[c] - orig_u[c] * orig_u[c];
            ddot += err0 * orig_u[c];
            dsum += err0;
          }
          auto eval_cost = [&](size_t c, int v_cur, int v_new) {
            double dv = v_new - v_cur;
            double e_cur = v_cur - orig_u[c], e_new = v_new - orig_u[c];
            double d_sq_new = dsq + (double(v_new) * v_new - double(v_cur) * v_cur);
            double d_dot_new = ddot + dv * orig_u[c];
            double d_sum_new = dsum + dv;
            return (e_new * e_new - e_cur * e_cur)
                 + w_sq * (d_sq_new * d_sq_new - dsq * dsq)
                 + w_dot * (d_dot_new * d_dot_new - ddot * ddot)
                 + w_sum * (d_sum_new * d_sum_new - dsum * dsum)
                 + lam_g * (r_bits(v_new) - r_bits(v_cur));
          };
          auto snap_u = [&](size_t c, int vn_lo, int vn_hi) {
            int v = urow[c];
            int vn1 = (eval_cost(c, v, vn_hi) <= eval_cost(c, v, vn_lo)) ? vn_hi : vn_lo;
            if (vn1 == 0 && v != 0 && r_nnz <= 1) vn1 = vn_hi;
            int vn2 = (vn1 == vn_hi) ? vn_lo : vn_hi;
            if (vn2 == 0 && v != 0 && r_nnz <= 1) vn2 = vn_hi;
            int vn = v;
            urow[c] = int8_t(vn1);
            if (order_ok(j)) vn = vn1;
            else {
              urow[c] = int8_t(vn2);
              if (order_ok(j)) vn = vn2;
              else urow[c] = int8_t(v);
            }
            if (vn == 0 && v != 0) r_nnz--;
            double dv = vn - v;
            dsq += double(vn) * vn - double(v) * v;
            ddot += dv * orig_u[c];
            dsum += dv;
            urow[c] = int8_t(vn);
          };
          std::vector<size_t> c1, c35, c247;
          for (size_t c = 0; c < D; c++) {
            int a = std::abs(raw_u[c]);
            if (a == 1) c1.push_back(c);
            else if (a == 3 || a == 5 || a == 6) c35.push_back(c);
            else if (a == 2 || a == 4 || a == 7) c247.push_back(c);
          }
          for (size_t c : c1) snap_u(c, 0, 2 * ((raw_u[c] > 0) ? 1 : -1));
          for (size_t c : c35) {
            int s_u = (raw_u[c] > 0) ? 1 : -1, a = std::abs(raw_u[c]);
            int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
            snap_u(c, s_u * lo, s_u * hi);
          }
          if (lam_g == 0.0 && alpha_g == 1.0) {
            for (int pass = 0; pass < 4; pass++) {
              for (size_t c : c1) {
                int s_u = (raw_u[c] > 0) ? 1 : -1, v_cur = urow[c], v_alt = (v_cur == 0) ? (2 * s_u) : 0;
                if (eval_cost(c, v_cur, v_alt) < -1e-12) {
                  urow[c] = int8_t(v_alt);
                  if (order_ok(j)) {
                    double d_alt = v_alt - v_cur;
                    dsq += double(v_alt) * v_alt - double(v_cur) * v_cur;
                    ddot += d_alt * orig_u[c];
                    dsum += d_alt;
                  } else urow[c] = int8_t(v_cur);
                }
              }
              for (size_t c : c35) {
                int s_u = (raw_u[c] > 0) ? 1 : -1, a = std::abs(raw_u[c]), v_cur = urow[c];
                int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
                int v_alt = (v_cur == s_u * lo) ? (s_u * hi) : (s_u * lo);
                if (eval_cost(c, v_cur, v_alt) < -1e-12) {
                  urow[c] = int8_t(v_alt);
                  if (order_ok(j)) {
                    double d_alt = v_alt - v_cur;
                    dsq += double(v_alt) * v_alt - double(v_cur) * v_cur;
                    ddot += d_alt * orig_u[c];
                    dsum += d_alt;
                  } else urow[c] = int8_t(v_cur);
                }
              }
            }
          } else {
            auto try_multi_gup = [&](size_t c) {
              int a = std::abs(raw_u[c]);
              if (a == 0) return;
              int s_u = (raw_u[c] > 0) ? 1 : -1, v_cur = urow[c];
              int hi = (a <= 2) ? 2 : ((a <= 4) ? 4 : 7);
              int lo_min = (min_lv_35 && a >= 5) ? 2 : ((min_lv_35 == 2 && a >= 3) ? 2 : 0);
              const int levels[4] = {0, 2 * s_u, 4 * s_u, 7 * s_u};
              int best_v = v_cur;
              double best_cost = -1e-12;
              for (int lv : levels) {
                if (std::abs(lv) > hi || std::abs(lv) < lo_min || lv == v_cur) continue;
                if (lv == 0 && v_cur != 0 && r_nnz <= 1) continue;
                double c_lv = eval_cost(c, v_cur, lv);
                if (c_lv < best_cost) {
                  urow[c] = int8_t(lv);
                  if (order_ok(j)) { best_cost = c_lv; best_v = lv; }
                  urow[c] = int8_t(v_cur);
                }
              }
              if (best_v != v_cur) {
                if (best_v == 0 && v_cur != 0) r_nnz--;
                else if (best_v != 0 && v_cur == 0) r_nnz++;
                double d_alt = best_v - v_cur;
                dsq += double(best_v) * best_v - double(v_cur) * v_cur;
                ddot += d_alt * orig_u[c];
                dsum += d_alt;
                urow[c] = int8_t(best_v);
              }
            };
            for (int pass = 0; pass < 4; pass++) {
              for (size_t c : c1) try_multi_gup(c);
              for (size_t c : c35) try_multi_gup(c);
              for (size_t c : c247) try_multi_gup(c);
            }
          }
        }
        std::vector<int> idx(N_bot);
        for (size_t j = 0; j < N_bot; j++) idx[j] = int(j);
        std::sort(idx.begin(), idx.end(), [&](int x, int y) {
          return std::memcmp(&es[i_up_q].data[size_t(x) * D], &es[i_up_q].data[size_t(y) * D], D) < 0;
        });
        std::vector<int> perm(N_bot);
        for (size_t j = 0; j < N_bot; j++) {
          uint32_t d = bot_digs[j];
          perm[j] = idx[d];
          idx.erase(idx.begin() + d);
        }
        for (const auto& G : gs) {
          if (G.key == i_up_q) {
            apply_perm(es, G, perm);
            break;
          }
        }
        std::vector<double> old_u = u_orig;
        for (size_t j = 0; j < N_bot; j++)
          std::memcpy(&u_orig[j * D], &old_u[size_t(perm[j]) * D], D * sizeof(double));
        const uint16_t* up_sc = reinterpret_cast<const uint16_t*>(es[i_up_s].data.data());
        std::vector<double> u_new(N_bot * D, 0.0), bot_imp(N_bot, 0.0), bot_orig_imp(N_bot, 0.0);
        for (size_t j = 0; j < N_bot; j++) {
          double sc = bf16_val(up_sc[j]);
          const int8_t* urow = reinterpret_cast<const int8_t*>(&es[i_up_q].data[j * D]);
          for (size_t d = 0; d < D; d++) {
            double v = sc * urow[d];
            u_new[j * D + d] = v;
            bot_imp[j] += v * v;
            bot_orig_imp[j] += u_orig[j * D + d] * u_orig[j * D + d];
          }
        }
        std::vector<double> cov(N_bot * N_bot, 0.0), cov_cross(N_bot * N_bot, 0.0);
        for (size_t i = 0; i < N_bot; i++) {
          for (size_t j = 0; j < N_bot; j++) {
            double s_nn = 0.0, s_no = 0.0;
            for (size_t d = 0; d < D; d++) {
              s_nn += u_new[i * D + d] * u_new[j * D + d];
              s_no += u_new[i * D + d] * u_orig[j * D + d];
            }
            if (i == j) {
              s_nn += 1e-2 * bot_imp[i];
              s_no += 1e-2 * std::sqrt(bot_imp[i] * bot_orig_imp[i]);
            }
            cov[i * N_bot + j] = s_nn;
            cov_cross[i * N_bot + j] = s_no;
          }
        }
        double mean_cov = 0.0;
        for (size_t j = 0; j < N_bot; j++) mean_cov += cov[j * N_bot + j];
        mean_cov /= double(N_bot);
        for (size_t r = 0; r < D; r++) {
          int8_t* drow = reinterpret_cast<int8_t*>(&es[i_dn_q].data[r * N_bot]);
          double inv_alpha_dn = 1.0;
          double alpha_g_dn = 1.0 + 1.0 * (alpha_g - 1.0);
          double lam_g_dn = 1.0 * lam_g;
          if (alpha_g_dn > 1.0) {
            float sc_orig = bf16_val(dn_sc_mut[r]);
            uint16_t b_new = f32_to_bf16_4bit(float(sc_orig * alpha_g_dn));
            float sc_new = bf16_val(b_new);
            if (sc_new > 0.0f && sc_orig > 0.0f) inv_alpha_dn = double(sc_orig) / double(sc_new);
            dn_sc_mut[r] = b_new;
          }
          std::vector<int> orig_w(N_bot);
          int r_nnz = 0;
          for (size_t j = 0; j < N_bot; j++) { orig_w[j] = drow[j]; if (drow[j]) r_nnz++; }
          if (r_nnz == 0) continue;
          std::vector<double> g(N_bot, 0.0);
          for (size_t i = 0; i < N_bot; i++) {
            double s_g = 0.0;
            for (size_t j = 0; j < N_bot; j++)
              s_g += (cov[i * N_bot + j] - inv_alpha_dn * cov_cross[i * N_bot + j]) * orig_w[j];
            g[i] = s_g;
          }
          auto snap_col = [&](size_t c, int vn_lo, int vn_hi) {
            int v = drow[c];
            double d_lo = vn_lo - v, d_hi = vn_hi - v, c_cc = cov[c * N_bot + c];
            double cost_hi = 2.0 * d_hi * g[c] + d_hi * d_hi * c_cc + lam_g_dn * mean_cov * r_bits(vn_hi);
            double cost_lo = 2.0 * d_lo * g[c] + d_lo * d_lo * c_cc + lam_g_dn * mean_cov * r_bits(vn_lo);
            int vn = (cost_hi <= cost_lo) ? vn_hi : vn_lo;
            if (vn == 0 && v != 0 && r_nnz <= 1) vn = vn_hi;
            if (vn == 0 && v != 0) r_nnz--;
            double dv = vn - v;
            drow[c] = int8_t(vn);
            const double* cov_c = &cov[c * N_bot];
            for (size_t j = 0; j < N_bot; j++) g[j] += dv * cov_c[j];
          };
          std::vector<size_t> c1, c35, c247;
          for (size_t c = 0; c < N_bot; c++) {
            int a = std::abs(int(drow[c]));
            if (a == 1) c1.push_back(c);
            else if (a == 3 || a == 5 || a == 6) c35.push_back(c);
            else if (a == 2 || a == 4 || a == 7) c247.push_back(c);
          }
          auto cmp_c = [&](size_t a, size_t b) { return bot_imp[a] < bot_imp[b]; };
          std::sort(c1.begin(), c1.end(), cmp_c);
          std::sort(c35.begin(), c35.end(), cmp_c);
          std::sort(c247.begin(), c247.end(), cmp_c);
          for (size_t c : c1) snap_col(c, 0, 2 * ((drow[c] > 0) ? 1 : -1));
          for (size_t c : c35) {
            int s = (drow[c] > 0) ? 1 : -1, a = std::abs(int(drow[c]));
            int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
            snap_col(c, s * lo, s * hi);
          }
          if (lam_g == 0.0 && alpha_g == 1.0) {
            for (int pass = 0; pass < 4; pass++) {
              for (size_t c : c1) {
                int s = (orig_w[c] > 0) ? 1 : -1, v_cur = drow[c], v_alt = (v_cur == 0) ? (2 * s) : 0;
                double d_alt = v_alt - v_cur;
                if (2.0 * d_alt * g[c] + d_alt * d_alt * cov[c * N_bot + c] < -1e-12) {
                  drow[c] = int8_t(v_alt);
                  const double* cov_c = &cov[c * N_bot];
                  for (size_t j = 0; j < N_bot; j++) g[j] += d_alt * cov_c[j];
                }
              }
              for (size_t c : c35) {
                int s = (orig_w[c] > 0) ? 1 : -1, a = std::abs(orig_w[c]), v_cur = drow[c];
                int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
                int v_alt = (v_cur == s * lo) ? (s * hi) : (s * lo);
                double d_alt = v_alt - v_cur;
                if (2.0 * d_alt * g[c] + d_alt * d_alt * cov[c * N_bot + c] < -1e-12) {
                  drow[c] = int8_t(v_alt);
                  const double* cov_c = &cov[c * N_bot];
                  for (size_t j = 0; j < N_bot; j++) g[j] += d_alt * cov_c[j];
                }
              }
            }
          } else {
            auto try_multi_gdn = [&](size_t c) {
              int a = std::abs(orig_w[c]);
              if (a == 0) return;
              int s = (orig_w[c] > 0) ? 1 : -1, v_cur = drow[c];
              int hi = (a <= 2) ? 2 : ((a <= 4) ? 4 : 7);
              int lo_min = (min_lv_35 && a >= 5) ? 2 : ((min_lv_35 == 2 && a >= 3) ? 2 : 0);
              const int levels[4] = {0, 2 * s, 4 * s, 7 * s};
              int best_v = v_cur;
              double best_cost = -1e-12;
              for (int lv : levels) {
                if (std::abs(lv) > hi || std::abs(lv) < lo_min || lv == v_cur) continue;
                if (lv == 0 && v_cur != 0 && r_nnz <= 1) continue;
                double d_alt = lv - v_cur;
                double c_lv = 2.0 * d_alt * g[c] + d_alt * d_alt * cov[c * N_bot + c] + lam_g_dn * mean_cov * (r_bits(lv) - r_bits(v_cur));
                if (c_lv < best_cost) { best_cost = c_lv; best_v = lv; }
              }
              if (best_v != v_cur) {
                if (best_v == 0 && v_cur != 0) r_nnz--;
                else if (best_v != 0 && v_cur == 0) r_nnz++;
                double d_alt = best_v - v_cur;
                drow[c] = int8_t(best_v);
                const double* cov_c = &cov[c * N_bot];
                for (size_t j = 0; j < N_bot; j++) g[j] += d_alt * cov_c[j];
              }
            };
            for (int pass = 0; pass < 4; pass++) {
              for (size_t c : c1) try_multi_gdn(c);
              for (size_t c : c35) try_multi_gdn(c);
              for (size_t c : c247) try_multi_gdn(c);
            }
          }
        }
      }
    }

    // QAT snapping for query_projection and key_projection in blocks >= qk_min_blk
    int qk_min_blk = (argc >= 8) ? std::atoi(argv[7]) : 0;
    int o_min_blk = (argc >= 9) ? std::atoi(argv[8]) : 0;
    int v_min_blk = (argc >= 10) ? std::atoi(argv[9]) : 0;
    #pragma omp parallel for schedule(dynamic)
    for (int blk = qk_min_blk; blk < 16; blk++) {
      std::string pref = "blocks." + std::to_string(blk) + ".attention.";
      int i_qq = -1, i_qs = -1, i_kq = -1, i_ks = -1;
      for (size_t i = 0; i <= last; i++) {
        if (es[i].name == pref + "query_projection.weight.q") i_qq = int(i);
        if (es[i].name == pref + "query_projection.weight.scale") i_qs = int(i);
        if (es[i].name == pref + "key_projection.weight.q") i_kq = int(i);
        if (es[i].name == pref + "key_projection.weight.scale") i_ks = int(i);
      }
      if (i_qq < 0 || i_qs < 0 || i_kq < 0 || i_ks < 0) continue;
      const size_t D = 192;
      const size_t Rk = es[i_kq].rows();
      std::vector<int8_t> k_orig_all(es[i_kq].data.begin(), es[i_kq].data.end());
      std::vector<uint16_t> k_sc_orig_all(Rk);
      std::memcpy(k_sc_orig_all.data(), es[i_ks].data.data(), Rk * sizeof(uint16_t));
      for (int idx_q : {i_kq, i_qq}) {
        int idx_s = (idx_q == i_kq) ? i_ks : i_qs;
        double cur_lam_qk = (idx_q == i_qq) ? (1.0 * lam_qk) : lam_qk;
        double cur_alpha_qk = (idx_q == i_qq) ? (1.0 + 1.0 * (alpha_qk - 1.0)) : alpha_qk;
        uint16_t* sc_mut = reinterpret_cast<uint16_t*>(es[idx_s].data.data());
        size_t R = es[idx_q].rows();
        for (size_t r = 0; r < R; r++) {
          int8_t* row = reinterpret_cast<int8_t*>(&es[idx_q].data[r * D]);
          int r_nnz = 0;
          for (size_t d = 0; d < D; d++) if (row[d]) r_nnz++;
          if (r_nnz == 0) continue;
          double inv_alpha = 1.0;
          if (cur_alpha_qk > 1.0) {
            float sc_orig = bf16_val(sc_mut[r]);
            uint16_t b_new = f32_to_bf16_4bit(float(sc_orig * cur_alpha_qk));
            float sc_new = bf16_val(b_new);
            if (sc_new > 0.0f && sc_orig > 0.0f) inv_alpha = double(sc_orig) / double(sc_new);
            sc_mut[r] = b_new;
          }
          std::vector<int> raw_u(D);
          std::vector<double> orig_u(D);
          double sq_orig = 0.0;
          for (size_t d = 0; d < D; d++) {
            raw_u[d] = row[d];
            orig_u[d] = double(row[d]) * inv_alpha;
            sq_orig += orig_u[d] * orig_u[d];
          }
          if (sq_orig <= 0.0) continue;
          size_t rk = (Rk == 64) ? (r % 64) : r;
          const int8_t* k_new_row = reinterpret_cast<const int8_t*>(&es[i_kq].data[rk * D]);
          const int8_t* k_old_row = &k_orig_all[rk * D];
          double k_ratio = 1.0;
          if (idx_q == i_qq && alpha_qk > 1.0) {
            float ks_old = bf16_val(k_sc_orig_all[rk]);
            float ks_new = bf16_val(reinterpret_cast<const uint16_t*>(es[i_ks].data.data())[rk]);
            if (ks_new > 0.0f && ks_old > 0.0f) k_ratio = double(ks_old) / double(ks_new);
          }
          double sq_k = 0.0;
          double dpk0 = 0.0;
          if (idx_q == i_qq) {
            for (size_t d = 0; d < D; d++) {
              sq_k += double(k_new_row[d]) * k_new_row[d];
              dpk0 += double(raw_u[d]) * double(k_new_row[d]) - orig_u[d] * (k_ratio * double(k_old_row[d]));
            }
          }
          double w_sq = w_sq_qk_k / sq_orig, w_dot = w_sq_qk_k / sq_orig, w_sum = 2.0 / double(D);
          double w_pk = (idx_q == i_qq && sq_k > 0.0) ? (0.5 * w_sq_qk_k / sq_k) : 0.0;
          double dsq = 0.0, ddot = 0.0, dsum = 0.0, dpk = dpk0;
          for (size_t c = 0; c < D; c++) {
            double err0 = double(row[c]) - orig_u[c];
            dsq += double(row[c]) * row[c] - orig_u[c] * orig_u[c];
            ddot += err0 * orig_u[c];
            dsum += err0;
          }
          auto eval_cost = [&](size_t c, int v_cur, int v_new) {
            double dv = v_new - v_cur;
            double e_cur = v_cur - orig_u[c], e_new = v_new - orig_u[c];
            double d_sq_new = dsq + (double(v_new) * v_new - double(v_cur) * v_cur);
            double d_dot_new = ddot + dv * orig_u[c];
            double d_sum_new = dsum + dv;
            double d_pk_new = dpk + ((idx_q == i_qq) ? dv * double(k_new_row[c]) : 0.0);
            return (e_new * e_new - e_cur * e_cur)
                 + w_sq * (d_sq_new * d_sq_new - dsq * dsq)
                 + w_dot * (d_dot_new * d_dot_new - ddot * ddot)
                 + w_sum * (d_sum_new * d_sum_new - dsum * dsum)
                 + w_pk * (d_pk_new * d_pk_new - dpk * dpk)
                 + cur_lam_qk * (r_bits(v_new) - r_bits(v_cur));
          };
          auto snap_u = [&](size_t c, int vn_lo, int vn_hi) {
            int v = row[c];
            int vn = (eval_cost(c, v, vn_hi) <= eval_cost(c, v, vn_lo)) ? vn_hi : vn_lo;
            if (vn == 0 && v != 0 && r_nnz <= 1) vn = vn_hi;
            if (vn == 0 && v != 0) r_nnz--;
            double dv = vn - v;
            dsq += double(vn) * vn - double(v) * v;
            ddot += dv * orig_u[c];
            dsum += dv;
            if (idx_q == i_qq) dpk += dv * double(k_new_row[c]);
            row[c] = int8_t(vn);
          };
          std::vector<size_t> c1, c35, c247;
          for (size_t c = 0; c < D; c++) {
            int a = std::abs(raw_u[c]);
            if (a == 1) c1.push_back(c);
            else if (a == 3 || a == 5 || a == 6) c35.push_back(c);
            else if (a == 2 || a == 4 || a == 7) c247.push_back(c);
          }
          for (size_t c : c1) snap_u(c, 0, 2 * ((raw_u[c] > 0) ? 1 : -1));
          for (size_t c : c35) {
            int s_u = (raw_u[c] > 0) ? 1 : -1, a = std::abs(raw_u[c]);
            int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
            snap_u(c, s_u * lo, s_u * hi);
          }
          if (lam_qk == 0.0 && alpha_qk == 1.0) {
            for (int pass = 0; pass < 4; pass++) {
              for (size_t c : c1) {
                int s_u = (raw_u[c] > 0) ? 1 : -1, v_cur = row[c], v_alt = (v_cur == 0) ? (2 * s_u) : 0;
                if (eval_cost(c, v_cur, v_alt) < -1e-12) {
                  double d_alt = v_alt - v_cur;
                  dsq += double(v_alt) * v_alt - double(v_cur) * v_cur;
                  ddot += d_alt * orig_u[c];
                  dsum += d_alt;
                  if (idx_q == i_qq) dpk += d_alt * double(k_new_row[c]);
                  row[c] = int8_t(v_alt);
                }
              }
              for (size_t c : c35) {
                int s_u = (raw_u[c] > 0) ? 1 : -1, a = std::abs(raw_u[c]), v_cur = row[c];
                int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
                int v_alt = (v_cur == s_u * lo) ? (s_u * hi) : (s_u * lo);
                if (eval_cost(c, v_cur, v_alt) < -1e-12) {
                  double d_alt = v_alt - v_cur;
                  dsq += double(v_alt) * v_alt - double(v_cur) * v_cur;
                  ddot += d_alt * orig_u[c];
                  dsum += d_alt;
                  if (idx_q == i_qq) dpk += d_alt * double(k_new_row[c]);
                  row[c] = int8_t(v_alt);
                }
              }
            }
          } else {
            auto try_multi_qk = [&](size_t c) {
              int a = std::abs(raw_u[c]);
              if (a == 0) return;
              int s_u = (raw_u[c] > 0) ? 1 : -1, v_cur = row[c];
              int hi = (a <= 2) ? 2 : ((a <= 4) ? 4 : 7);
              int lo_min = (min_lv_35 && a >= 5) ? 2 : ((min_lv_35 == 2 && a >= 3) ? 2 : 0);
              const int levels[4] = {0, 2 * s_u, 4 * s_u, 7 * s_u};
              int best_v = v_cur;
              double best_cost = -1e-12;
              for (int lv : levels) {
                if (std::abs(lv) > hi || std::abs(lv) < lo_min || lv == v_cur) continue;
                if (lv == 0 && v_cur != 0 && r_nnz <= 1) continue;
                double c_lv = eval_cost(c, v_cur, lv);
                if (c_lv < best_cost) { best_cost = c_lv; best_v = lv; }
              }
              if (best_v != v_cur) {
                if (best_v == 0 && v_cur != 0) r_nnz--;
                else if (best_v != 0 && v_cur == 0) r_nnz++;
                double d_alt = best_v - v_cur;
                dsq += double(best_v) * best_v - double(v_cur) * v_cur;
                ddot += d_alt * orig_u[c];
                dsum += d_alt;
                if (idx_q == i_qq) dpk += d_alt * double(k_new_row[c]);
                row[c] = int8_t(best_v);
              }
            };
            for (int pass = 0; pass < 4; pass++) {
              for (size_t c : c1) try_multi_qk(c);
              for (size_t c : c35) try_multi_qk(c);
              for (size_t c : c247) try_multi_qk(c);
            }
          }
        }
      }
    }
    // Coupled QAT snapping for value_projection and output_projection in blocks >= vo_min_blk
    #pragma omp parallel for schedule(dynamic)
    for (int blk = std::min(o_min_blk, v_min_blk); blk < 16; blk++) {
      std::string pref = "blocks." + std::to_string(blk) + ".attention.";
      int i_vq = -1, i_vs = -1, i_oq = -1, i_os = -1;
      for (size_t i = 0; i <= last; i++) {
        if (es[i].name == pref + "value_projection.weight.q") i_vq = int(i);
        if (es[i].name == pref + "value_projection.weight.scale") i_vs = int(i);
        if (es[i].name == pref + "output_projection.weight.q") i_oq = int(i);
        if (es[i].name == pref + "output_projection.weight.scale") i_os = int(i);
      }
      if (i_vq < 0 || i_vs < 0 || i_oq < 0 || i_os < 0) continue;
      const size_t Rv = es[i_vq].rows(), D = 192;
      uint16_t* v_sc_mut = reinterpret_cast<uint16_t*>(es[i_vs].data.data());
      uint16_t* o_sc_mut = reinterpret_cast<uint16_t*>(es[i_os].data.data());
      std::vector<uint32_t> v_digs(64, 0);
      for (size_t k = 0; k < 64; k++) {
        uint32_t c = 0;
        for (size_t j = k + 1; j < 64; j++)
          c += std::memcmp(&es[i_vq].data[j * D], &es[i_vq].data[k * D], D) < 0;
        v_digs[k] = c;
      }
      int i_vc = -1, i_wn = -1;
      for (size_t i = 0; i <= last; i++) {
        if (es[i].name == pref + "value_convolution.weight") i_vc = int(i);
        if (es[i].name == pref + "output_fused_norm_gate.weight") i_wn = int(i);
      }
      std::vector<double> ch_mult(Rv, 1.0);
      if (i_vc >= 0 && i_wn >= 0) {
        const float* vc = reinterpret_cast<const float*>(es[i_vc].data.data());
        const float* wn = reinterpret_cast<const float*>(es[i_wn].data.data());
        for (size_t j = 0; j < Rv; j++) {
          double csq = 0.0;
          for (size_t t = 0; t < 4; t++) csq += double(vc[j * 4 + t]) * vc[j * 4 + t];
          ch_mult[j] = std::sqrt(csq + 1e-6) * std::abs(double(wn[j % 64]));
        }
      }
      std::vector<double> v_orig(Rv * D, 0.0);
      for (size_t j = 0; j < Rv; j++) {
        double sc = bf16_val(v_sc_mut[j]) * ch_mult[j];
        const int8_t* vrow = reinterpret_cast<const int8_t*>(&es[i_vq].data[j * D]);
        for (size_t d = 0; d < D; d++) v_orig[j * D + d] = sc * vrow[d];
      }
      auto order_ok = [&](size_t j) {
        if (j >= 64) return true;
        const uint8_t* cur = &es[i_vq].data[j * D];
        for (size_t o = 0; o < 64; o++)
          if (o != j && std::memcmp(&es[i_vq].data[o * D], cur, D) == 0) return false;
        return true;
      };
      if (blk >= v_min_blk) {
      for (size_t j = 0; j < Rv; j++) {
        int8_t* vrow = reinterpret_cast<int8_t*>(&es[i_vq].data[j * D]);
        int r_nnz = 0;
        for (size_t d = 0; d < D; d++) if (vrow[d]) r_nnz++;
        if (r_nnz == 0) continue;
        double inv_alpha = 1.0;
        if (alpha_v > 1.0) {
          float sc_orig = bf16_val(v_sc_mut[j]);
          uint16_t b_new = f32_to_bf16_4bit(float(sc_orig * alpha_v));
          float sc_new = bf16_val(b_new);
          if (sc_new > 0.0f && sc_orig > 0.0f) inv_alpha = double(sc_orig) / double(sc_new);
          v_sc_mut[j] = b_new;
        }
        std::vector<int> raw_u(D);
        std::vector<double> orig_u(D);
        double sq_orig = 0.0;
        for (size_t d = 0; d < D; d++) {
          raw_u[d] = vrow[d];
          orig_u[d] = double(vrow[d]) * inv_alpha;
          sq_orig += orig_u[d] * orig_u[d];
        }
        if (sq_orig <= 0.0) continue;
        double w_sq = w_sq_v_k / sq_orig, w_dot = w_sq_v_k / sq_orig, w_sum = 2.0 / double(D);
        double dsq = 0.0, ddot = 0.0, dsum = 0.0;
        for (size_t c = 0; c < D; c++) {
          double err0 = double(vrow[c]) - orig_u[c];
          dsq += double(vrow[c]) * vrow[c] - orig_u[c] * orig_u[c];
          ddot += err0 * orig_u[c];
          dsum += err0;
        }
        auto eval_cost = [&](size_t c, int v_cur, int v_new) {
          double dv = v_new - v_cur;
          double e_cur = v_cur - orig_u[c], e_new = v_new - orig_u[c];
          double d_sq_new = dsq + (double(v_new) * v_new - double(v_cur) * v_cur);
          double d_dot_new = ddot + dv * orig_u[c];
          double d_sum_new = dsum + dv;
          return (e_new * e_new - e_cur * e_cur)
               + w_sq * (d_sq_new * d_sq_new - dsq * dsq)
               + w_dot * (d_dot_new * d_dot_new - ddot * ddot)
               + w_sum * (d_sum_new * d_sum_new - dsum * dsum)
               + lam_v * (r_bits(v_new) - r_bits(v_cur));
        };
        auto snap_u = [&](size_t c, int vn_lo, int vn_hi) {
          int v = vrow[c];
          int vn1 = (eval_cost(c, v, vn_hi) <= eval_cost(c, v, vn_lo)) ? vn_hi : vn_lo;
          if (vn1 == 0 && v != 0 && r_nnz <= 1) vn1 = vn_hi;
          int vn2 = (vn1 == vn_hi) ? vn_lo : vn_hi;
          if (vn2 == 0 && v != 0 && r_nnz <= 1) vn2 = vn_hi;
          int vn = v;
          vrow[c] = int8_t(vn1);
          if (order_ok(j)) vn = vn1;
          else {
            vrow[c] = int8_t(vn2);
            if (order_ok(j)) vn = vn2;
            else vrow[c] = int8_t(v);
          }
          if (vn == 0 && v != 0) r_nnz--;
          double dv = vn - v;
          dsq += double(vn) * vn - double(v) * v;
          ddot += dv * orig_u[c];
          dsum += dv;
          vrow[c] = int8_t(vn);
        };
        std::vector<size_t> c1, c35, c247;
        for (size_t c = 0; c < D; c++) {
          int a = std::abs(raw_u[c]);
          if (a == 1) c1.push_back(c);
          else if (a == 3 || a == 5 || a == 6) c35.push_back(c);
          else if (a == 2 || a == 4 || a == 7) c247.push_back(c);
        }
        for (size_t c : c1) snap_u(c, 0, 2 * ((raw_u[c] > 0) ? 1 : -1));
        for (size_t c : c35) {
          int s_u = (raw_u[c] > 0) ? 1 : -1, a = std::abs(raw_u[c]);
          int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
          snap_u(c, s_u * lo, s_u * hi);
        }
        if (lam_v == 0.0 && alpha_v == 1.0) {
          for (int pass = 0; pass < 4; pass++) {
            for (size_t c : c1) {
              int s_u = (raw_u[c] > 0) ? 1 : -1, v_cur = vrow[c], v_alt = (v_cur == 0) ? (2 * s_u) : 0;
              if (eval_cost(c, v_cur, v_alt) < -1e-12) {
                vrow[c] = int8_t(v_alt);
                if (order_ok(j)) {
                  double d_alt = v_alt - v_cur;
                  dsq += double(v_alt) * v_alt - double(v_cur) * v_cur;
                  ddot += d_alt * orig_u[c];
                  dsum += d_alt;
                } else vrow[c] = int8_t(v_cur);
              }
            }
            for (size_t c : c35) {
              int s_u = (raw_u[c] > 0) ? 1 : -1, a = std::abs(raw_u[c]), v_cur = vrow[c];
              int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
              int v_alt = (v_cur == s_u * lo) ? (s_u * hi) : (s_u * lo);
              if (eval_cost(c, v_cur, v_alt) < -1e-12) {
                vrow[c] = int8_t(v_alt);
                if (order_ok(j)) {
                  double d_alt = v_alt - v_cur;
                  dsq += double(v_alt) * v_alt - double(v_cur) * v_cur;
                  ddot += d_alt * orig_u[c];
                  dsum += d_alt;
                } else vrow[c] = int8_t(v_cur);
              }
            }
          }
        } else {
          auto try_multi_v = [&](size_t c) {
            int a = std::abs(raw_u[c]);
            if (a == 0) return;
            int s_u = (raw_u[c] > 0) ? 1 : -1, v_cur = vrow[c];
            int hi = (a <= 2) ? 2 : ((a <= 4) ? 4 : 7);
            int lo_min = (min_lv_35 && a >= 5) ? 2 : ((min_lv_35 == 2 && a >= 3) ? 2 : 0);
            const int levels[4] = {0, 2 * s_u, 4 * s_u, 7 * s_u};
            int best_v = v_cur;
            double best_cost = -1e-12;
            for (int lv : levels) {
              if (std::abs(lv) > hi || std::abs(lv) < lo_min || lv == v_cur) continue;
              if (lv == 0 && v_cur != 0 && r_nnz <= 1) continue;
              double c_lv = eval_cost(c, v_cur, lv);
              if (c_lv < best_cost) {
                vrow[c] = int8_t(lv);
                if (order_ok(j)) { best_cost = c_lv; best_v = lv; }
                vrow[c] = int8_t(v_cur);
              }
            }
            if (best_v != v_cur) {
              if (best_v == 0 && v_cur != 0) r_nnz--;
              else if (best_v != 0 && v_cur == 0) r_nnz++;
              double d_alt = best_v - v_cur;
              dsq += double(best_v) * best_v - double(v_cur) * v_cur;
              ddot += d_alt * orig_u[c];
              dsum += d_alt;
              vrow[c] = int8_t(best_v);
            }
          };
          for (int pass = 0; pass < 4; pass++) {
            for (size_t c : c1) try_multi_v(c);
            for (size_t c : c35) try_multi_v(c);
            for (size_t c : c247) try_multi_v(c);
          }
        }
      }
      std::vector<int> idx(64);
      for (size_t j = 0; j < 64; j++) idx[j] = int(j);
      std::sort(idx.begin(), idx.end(), [&](int x, int y) {
        return std::memcmp(&es[i_vq].data[size_t(x) * D], &es[i_vq].data[size_t(y) * D], D) < 0;
      });
      std::vector<int> perm(64);
      for (size_t j = 0; j < 64; j++) {
        uint32_t d = v_digs[j];
        perm[j] = idx[d];
        idx.erase(idx.begin() + d);
      }
      for (const auto& G : gs) {
        if (G.key == i_vq && G.r0 == 0) {
          apply_perm(es, G, perm);
          break;
        }
      }
      std::vector<double> old_v = v_orig;
      for (size_t h = 0; h < Rv / 64; h++)
        for (size_t j = 0; j < 64; j++)
          std::memcpy(&v_orig[(64 * h + j) * D], &old_v[(64 * h + size_t(perm[j])) * D], D * sizeof(double));
      }
      const uint16_t* v_sc = reinterpret_cast<const uint16_t*>(es[i_vs].data.data());
      std::vector<double> v_new(Rv * D, 0.0);
      for (size_t j = 0; j < Rv; j++) {
        double sc = bf16_val(v_sc[j]) * ch_mult[j];
        const int8_t* vrow = reinterpret_cast<const int8_t*>(&es[i_vq].data[j * D]);
        for (size_t d = 0; d < D; d++) v_new[j * D + d] = sc * vrow[d];
      }
      std::vector<double> cov(D * D, 0.0), cov_cross(D * D, 0.0), ch_imp(D, 0.0);
      std::vector<double> nrm_new(Rv, 0.0), nrm_orig(Rv, 0.0);
      for (size_t j = 0; j < Rv; j++) {
        double sn = 0.0, so = 0.0;
        for (size_t d = 0; d < D; d++) {
          sn += v_new[j * D + d] * v_new[j * D + d];
          so += v_orig[j * D + d] * v_orig[j * D + d];
        }
        nrm_new[j] = std::sqrt(sn + 1e-12);
        nrm_orig[j] = std::sqrt(so + 1e-12);
      }
      for (size_t ci = 0; ci < D; ci++) {
        size_t vi = (Rv == 64) ? (ci % 64) : ci;
        const double* vni = &v_new[vi * D];
        double ni = nrm_new[vi];
        ch_imp[ci] = ni * ni;
        for (size_t cj = 0; cj < D; cj++) {
          size_t vj = (Rv == 64) ? (cj % 64) : cj;
          const double* vnj = &v_new[vj * D];
          const double* voj = &v_orig[vj * D];
          double nj = nrm_new[vj], noj = nrm_orig[vj];
          double dot_nn = 0.0, dot_no = 0.0;
          for (size_t d = 0; d < D; d++) {
            dot_nn += vni[d] * vnj[d];
            dot_no += vni[d] * voj[d];
          }
          auto f_kern = [&](double rho) {
            if (Rv == 64) return rho; // linear V in window attention
            rho = std::max(-1.0, std::min(1.0, rho));
            double th = std::acos(rho);
            double j1 = (std::sin(th) + (kPi - th) * rho) / kPi;
            return 0.5 * rho + 0.5 * j1;
          };
          double h_scale = (ci / 64 == cj / 64) ? 1.0 : ((Rv == 64) ? 0.25 : 0.10);
          double eff_noj = (Rv == 192 && (lam_v > 0.0 || alpha_v > 1.0 || lam_o > 0.0 || alpha_o > 1.0)) ? nj : noj;
          double s_nn = ni * nj * f_kern(dot_nn / (ni * nj)) * h_scale;
          double s_no = ni * eff_noj * f_kern(dot_no / (ni * noj)) * h_scale;
          if (ci == cj) {
            s_nn += ridge_o_k * ni * ni;
            s_no += ridge_o_k * ni * eff_noj;
          }
          cov[ci * D + cj] = s_nn;
          cov_cross[ci * D + cj] = s_no;
        }
      }
      double mean_cov = 0.0;
      for (size_t c = 0; c < D; c++) mean_cov += cov[c * D + c];
      mean_cov /= double(D);
      if (blk >= o_min_blk)
      for (size_t r = 0; r < D; r++) {
        int8_t* orow = reinterpret_cast<int8_t*>(&es[i_oq].data[r * D]);
        int r_nnz = 0;
        for (size_t j = 0; j < D; j++) if (orow[j]) r_nnz++;
        if (r_nnz == 0) continue;
        double inv_alpha_o = 1.0;
        if (alpha_o > 1.0) {
          float sc_orig = bf16_val(o_sc_mut[r]);
          uint16_t b_new = f32_to_bf16_4bit(float(sc_orig * alpha_o));
          float sc_new = bf16_val(b_new);
          if (sc_new > 0.0f && sc_orig > 0.0f) inv_alpha_o = double(sc_orig) / double(sc_new);
          o_sc_mut[r] = b_new;
        }
        std::vector<int> orig_w(D);
        for (size_t j = 0; j < D; j++) orig_w[j] = orow[j];
        std::vector<double> g(D, 0.0);
        for (size_t i = 0; i < D; i++) {
          double s_g = 0.0;
          for (size_t j = 0; j < D; j++)
            s_g += (cov[i * D + j] - inv_alpha_o * cov_cross[i * D + j]) * orig_w[j];
          g[i] = s_g;
        }
        auto snap_col = [&](size_t c, int vn_lo, int vn_hi) {
          int v = orow[c];
          double d_lo = vn_lo - v, d_hi = vn_hi - v, c_cc = cov[c * D + c];
          double cost_hi = 2.0 * d_hi * g[c] + d_hi * d_hi * c_cc + lam_o * mean_cov * r_bits(vn_hi);
          double cost_lo = 2.0 * d_lo * g[c] + d_lo * d_lo * c_cc + lam_o * mean_cov * r_bits(vn_lo);
          int vn = (cost_hi <= cost_lo) ? vn_hi : vn_lo;
          if (vn == 0 && v != 0 && r_nnz <= 1) vn = vn_hi;
          if (vn == 0 && v != 0) r_nnz--;
          double dv = vn - v;
          orow[c] = int8_t(vn);
          const double* cov_c = &cov[c * D];
          for (size_t j = 0; j < D; j++) g[j] += dv * cov_c[j];
        };
        std::vector<size_t> c1, c35, c247;
        for (size_t c = 0; c < D; c++) {
          int a = std::abs(orig_w[c]);
          if (a == 1) c1.push_back(c);
          else if (a == 3 || a == 5 || a == 6) c35.push_back(c);
          else if (a == 2 || a == 4 || a == 7) c247.push_back(c);
        }
        auto cmp_c = [&](size_t a, size_t b) { return ch_imp[a] < ch_imp[b]; };
        std::sort(c1.begin(), c1.end(), cmp_c);
        std::sort(c35.begin(), c35.end(), cmp_c);
        std::sort(c247.begin(), c247.end(), cmp_c);
        for (size_t c : c1) snap_col(c, 0, 2 * ((orig_w[c] > 0) ? 1 : -1));
        for (size_t c : c35) {
          int s = (orig_w[c] > 0) ? 1 : -1, a = std::abs(orig_w[c]);
          int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
          snap_col(c, s * lo, s * hi);
        }
        if (lam_o == 0.0 && alpha_o == 1.0) {
          for (int pass = 0; pass < 4; pass++) {
            for (size_t c : c1) {
              int s = (orig_w[c] > 0) ? 1 : -1, v_cur = orow[c], v_alt = (v_cur == 0) ? (2 * s) : 0;
              double d_alt = v_alt - v_cur;
              if (2.0 * d_alt * g[c] + d_alt * d_alt * cov[c * D + c] < -1e-12) {
                orow[c] = int8_t(v_alt);
                const double* cov_c = &cov[c * D];
                for (size_t j = 0; j < D; j++) g[j] += d_alt * cov_c[j];
              }
            }
            for (size_t c : c35) {
              int s = (orig_w[c] > 0) ? 1 : -1, a = std::abs(orig_w[c]), v_cur = orow[c];
              int lo = (a == 3) ? 2 : 4, hi = (a == 3) ? 4 : 7;
              int v_alt = (v_cur == s * lo) ? (s * hi) : (s * lo);
              double d_alt = v_alt - v_cur;
              if (2.0 * d_alt * g[c] + d_alt * d_alt * cov[c * D + c] < -1e-12) {
                orow[c] = int8_t(v_alt);
                const double* cov_c = &cov[c * D];
                for (size_t j = 0; j < D; j++) g[j] += d_alt * cov_c[j];
              }
            }
          }
        } else {
          auto try_multi_o = [&](size_t c) {
            int a = std::abs(orig_w[c]);
            if (a == 0) return;
            int s = (orig_w[c] > 0) ? 1 : -1, v_cur = orow[c];
            int hi = (a <= 2) ? 2 : ((a <= 4) ? 4 : 7);
            int lo_min = (min_lv_35 && a >= 5) ? 2 : ((min_lv_35 == 2 && a >= 3) ? 2 : 0);
            const int levels[4] = {0, 2 * s, 4 * s, 7 * s};
            int best_v = v_cur;
            double best_cost = -1e-12;
            double c_cc = cov[c * D + c];
            for (int lv : levels) {
              if (std::abs(lv) > hi || std::abs(lv) < lo_min || lv == v_cur) continue;
              if (lv == 0 && v_cur != 0 && r_nnz <= 1) continue;
              double d_alt = lv - v_cur;
              double c_lv = 2.0 * d_alt * g[c] + d_alt * d_alt * c_cc + lam_o * mean_cov * (r_bits(lv) - r_bits(v_cur));
              if (c_lv < best_cost) { best_cost = c_lv; best_v = lv; }
            }
            if (best_v != v_cur) {
              if (best_v == 0 && v_cur != 0) r_nnz--;
              else if (best_v != 0 && v_cur == 0) r_nnz++;
              double d_alt = best_v - v_cur;
              orow[c] = int8_t(best_v);
              const double* cov_c = &cov[c * D];
              for (size_t j = 0; j < D; j++) g[j] += d_alt * cov_c[j];
            }
          };
          for (int pass = 0; pass < 4; pass++) {
            for (size_t c : c1) try_multi_o(c);
            for (size_t c : c35) try_multi_o(c);
            for (size_t c : c247) try_multi_o(c);
          }
        }
      }
    }

    std::vector<uint8_t> enc = encodings(es, 5, true);
    BinEncoder EA;
    HeadCoder H;
    ValModels VA;
    for (size_t i = 0; i <= last; i++) {
      std::string cn = H.code(EA, es[i], uint8_t(enc[i] | keyflag[i]));
      encode_values(EA, VA, es[i], enc[i], cn, keyflag[i]);
    }
    std::vector<uint8_t> A = EA.finish();
    std::vector<uint8_t> out(8);
    std::memcpy(out.data(), "FX2TFWP5", 8);
    auto put = [&](uint32_t v) {
      auto b = u32le(v);
      out.insert(out.end(), b.begin(), b.end());
    };
    put(uint32_t(es.size()));
    put(uint32_t(last + 1));
    put(uint32_t(A.size() - 1));
    put(orig_K);
    out.insert(out.end(), A.begin() + 1, A.end());
    spit(out_path, out);
    std::vector<Ent> back = load_ents(out_path);
    bool same = back.size() == es.size();
    for (size_t i = 0; same && i < back.size(); i++)
      same = back[i].name == es[i].name && back[i].data == es[i].data;
    if (!same) {
      std::remove(out_path);
      die("STOP: qat-snap read back differs, output removed");
    }
    std::printf("qat-snap (min_blk=%d, up_min_blk=%d) -> %s: %zu B (%+ld B vs input, %+ld B in S), read back BIT IDENTICAL (%zu tensors)\n",
                min_blk, up_min_blk, out_path, out.size(), long(out.size()) - long(raw_in.size()),
                2 * (long(out.size()) - long(raw_in.size())), es.size());
    return 0;
  }
  std::fprintf(stderr, "usage: see the head comment of tools_src/tf1_np_codec.cpp\n");
  return 2;
}
