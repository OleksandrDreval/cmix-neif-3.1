// Modified in November 2026 by Oleksandr Dreval (https://github.com/OleksandrDreval)
// for cmix-neif-3.1 (https://github.com/OleksandrDreval/cmix-neif-3.1).
// SPDX-License-Identifier: GPL-3.0-only
#include "meld.h"
#include <string>
#include <ctime>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <malloc.h>

#include "preprocess/preprocessor.h"
#include "coder/encoder.h"
#include "coder/decoder.h"
#include "predictor.h"

#include "readalike_prepr/article_reorder.h"
#include "readalike_prepr/self_extract.h"
#include "readalike_prepr/phda9_preprocess.h"
#include "readalike_prepr/misc.h"
#include "readalike_prepr/komm_transform.h"
#include "r1_reorder_transform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "isa_floor.h"
#include "page_contexts.h"
#include "word_recency.h"
#include "simd_scan.h"
#include "invariants.h"
#include "huge_pages.h"
#include "mixer_lr.h"
#include "mixer_decay.h"
#include "fx_mixer_lr.h"
#include "mixer_bank.h"
#include "logit_table.h"
#include "mixer_weights.h"
#include "vocab_frozen.h"
#include "run_mode.h"

static_assert(54857 <= 65360,
    "exceeds the capacity of the code plan "
    "(dictionary.cpp kBoundary4 or kBoundary5): the words above it "
    "would get NO code word and be skipped silently.  Use a smaller "
    "dictionary, or open the fifth band and update EVERY reader of the "
    "band plan.");

static_assert(54857 <= run_mode::kBand4,
    "exceeds the bands the grammar automaton knows (kBand4): "
    "Idx3() would return a negative index for the fifth band and "
    "BuildMasks() would reject its second byte.  The grammar automaton "
    "needs a fifth band first.");

static_assert(54857 <= (80 + 3840 + 40960 + 20480),
    "exceeds the bands the PPMd warm-up knows "
    "(Code(), kBandKap4): its last branch would write a wrong code word for "
    "the fifth band and Material() would drop the words above.  Add the "
    "fifth band there first.");

// Run-time parameters: each one returns its compiled default.  The
// manifest writer is empty in this build (see its call in main()).
namespace run_options {
static double ParamD(const char* name, double def) {
  (void)name;
  return def;
}
static long ParamI(const char* name, long def) {
  double v = ParamD(name, (double)def);
  return (long)(v < 0 ? v - 0.5 : v + 0.5);
}
[[maybe_unused]] static void WriteManifest(const char* path, int argc, char** argv) {
  (void)path; (void)argc; (void)argv;
}
}

#include "attributes.h"

#include "run_mode.h"

#include "transformer.h"

#include "alias_context.h"
#include "alias_title_subst.h"
#include "author_block.h"
#include "metablock.h"
#include "census_block.h"
#include "computed_values.h"
static bool census_block_active = false;

#include "ip_pack.h"
#include "isbn_check.h"
#include "metablock_form.h"
#include "text_start_pack.h"
#include "rule_stages_file.h"
#include "ts_progress.h"


#include "no_args.h"


#include <malloc.h>
#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

namespace {
  constexpr long io_buffer_kb = 256;
  long progress_steps = 10000;  // progress ticks per run (they also drive the huge-page collapse)
}

#if !(1 && !AUX_BUILD)
static bool provide_dict_file(const char* src) {
  FILE* in = fopen(src, "rb");
  if (!in) return false;
  FILE* out = fopen(".dict", "wb");
  if (!out) { fclose(in); return false; }
  char buf[1 << 16];
  size_t n;
  bool ok = true;
  while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
    if (fwrite(buf, 1, n, out) != n) { ok = false; break; }
  }
  fclose(in);
  if (fclose(out) != 0) ok = false;
  return ok;
}
#endif

int Help() {
  msg_note(256);
  return -1;
}

size_t getFileSize(const char* path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    msg_note(170);
    return 0;
  }
  fseek(f, 0, SEEK_END);
  size_t output_size = ftell(f);
  fclose(f);
  return output_size;
}

void WriteStorageHeader(FILE* out, bool dictionary_used) {
  for (int i = 4; i >= 0; --i) {
    char c = 0;
    if (i == 4 && dictionary_used) c = 0x80;
    putc(c, out);
  }
}

void ClearOutput() {

}

#include "preprocess/dict_derive.h"
#include "huge_pages.h"
#include "vocab_frozen.h"
#include "console_progress.h"

#include "selfx_clean.h"

void WriteHeaderVocfix(unsigned long long length, bool dictionary_used,
    FILE* os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8*i);
    if (i == 4) {
      c &= 0x7F;
      if (dictionary_used) c |= 0x80;
      if (vocab_frozen::g_aux) c |= 0x40;
    }
    putc(c, os);
  }
}

void ReadHeaderVocfix(FILE* is, unsigned long long* length,
    bool* dictionary_used, std::vector<bool>* vocab) {
  *length = 0;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = getc(is);
    if (i == 0) {
      if (c&0x80) *dictionary_used = true;
      else *dictionary_used = false;
      c &= 0x7F;
      vocab_frozen::g_aux = (c & 0x40) != 0;
      c &= 0x3F;
    }
    *length += c;
  }
  vocab_frozen::FillVocab(vocab);
}

static inline bool AuthorBlockInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

static inline bool MetablockInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

static inline bool TimestampProgressInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

// The known bytes b[0..n) of the model side, coder and decoder alike: every
// model observes their bits, none is coded.  One function for both
// directions, so the hooks below need no coder object.
__attribute__((noinline)) static void ObserveKnownBytes(Predictor* p, const unsigned char* b, int n) {
  invariants::g_injected_byte = 1;
  for (int k = 0; k < n; ++k)
    for (int bit = 7; bit >= 0; --bit) p->ObserveKnownBit((b[k] >> bit) & 1);
  invariants::g_injected_byte = 0;
}

__attribute__((noinline)) static void CensusBlockByte(Predictor* p, unsigned char c) {
  if (MetablockInert()) return;
  if (metablock::mk.TaByte()) {
    // slot 3 is the pseudo article: at its end mark the automaton takes it
    if (metablock::mk.TaSlot() == 4 && metablock::mk.TaSlotEndet()) {
      size_t n = 0;
      const unsigned char* q = metablock::mk.TaPuffer(4, &n);
      census_block::aut.Slot3(q, n);
    }
    return;
  }
  census_block::aut.Observe(c);
  const unsigned char* zb = nullptr;
  const int zn = census_block::aut.TakeBlock(&zb);
  if (!zn) return;
  // of the census injections the transformer sees only the computed text (TfSieht)
  const bool tf = census_block::aut.TfSieht();
  if (!tf) timestamp_skip::Beginn();
  if (tf) tf_author_prompt::Beginn(zn);
  {
    const int kn = census_block::aut.EndeNach(zn);
    tf_copy_end::g_nach = tf ? kn : 0;
  }
  ObserveKnownBytes(p, zb, zn);
  tf_copy_end::g_nach = 0;
  if (tf) tf_author_prompt::Ende();
  if (!tf) timestamp_skip::Ende();
  (void)tf;
}

__attribute__((noinline)) static void TextStartHead(Predictor* p, int slot) {
  static const char kKopf2[] = "neif\tpre authors\n";
  static const char kKopf3[] = "neif\tpre census\n";
  static const char kKopf1[] = "neif\tpre dictionary\n";
  const char* k = slot == 2 ? kKopf2 : (slot == 4 ? kKopf3 : (slot == 1 ? kKopf1 : ""));
  const int n = (int)strlen(k);
  if (!n) return;
  tf_author_prompt::Beginn(n);
  ObserveKnownBytes(p, (const unsigned char*)k, n);
  tf_author_prompt::Ende();
}

static_assert(54857 == 54857, "must equal DICT_WORDS");
namespace fxcmv1 { int DictNachladen(); }
__attribute__((noinline)) static void DictArticleSlotEnd() {
  size_t n = 0;
  const unsigned char* b = metablock::mk.TaPuffer(1, &n);
  dict_article::Schreiben(b, n);
  const int w = fxcmv1::DictNachladen();
  if (dict_article::wrt) {
    run_mode::gram.Init(NULL, dict_article::strom, 0);
    run_mode::gram.ZoneSperren();
  }
  computed_values::g_dict_da = true;
  (void)w;
}

namespace dict_runs {
constexpr bool kRegel = 1 == 1;
// fixed bounds, checked against ./.dict by the coder (E551)
constexpr unsigned kTxt = dict_codes::kLaufTxt, kWort = dict_codes::kLaufWort;
struct RunState {                      // static (.bss), lives for the run
  size_t gez;                         // bytes of b followed
  unsigned nt, nw, nc;                // letters in txt, words finished, letters of the current word
  int np;                             // bytes of an open code word
  bool hinter;                        // a run mark in front of the current word
  unsigned char pend[3];              // the open code word
  unsigned char cur[kWort];           // the current word decoded so far
  unsigned ab[54857 + 1];       // start of word r in txt
  unsigned char txt[kTxt];            // the decoded words back to back
};
static RunState z;
static const unsigned char* Text(unsigned r, size_t* l) {
  *l = (r + 1 < z.nw ? z.ab[r + 1] : z.nt) - z.ab[r];
  return z.txt + z.ab[r];
}
// the rank intervals [lo[q], hi[q]] of the code words that begin with pre[0..k)
static int Intervalle(const unsigned char* pre, int k, unsigned* lo, unsigned* hi) {
  const unsigned b0 = pre[0];
  if (k == 1) {
    if (b0 <= 0xCF) { lo[0] = hi[0] = b0 - 0x80; return 1; }
    lo[0] = 80 + (b0 - 0xD0) * 80; hi[0] = lo[0] + 79;
    if (b0 < 0xF0) return 1;
    lo[1] = 3920 + (b0 - 0xF0) * 2560; hi[1] = lo[1] + 2559;
    lo[2] = 44880 + (b0 - 0xF0) * 1280; hi[2] = lo[2] + 1279;
    return 3;
  }
  const unsigned b1 = pre[1];
  if (b1 <= 0xCF) { lo[0] = hi[0] = 80 + (b0 - 0xD0) * 80 + (b1 - 0x80); return 1; }
  lo[0] = b1 <= 0xEF ? 3920 + ((b0 - 0xF0) * 32 + (b1 - 0xD0)) * 80
                     : 44880 + ((b0 - 0xF0) * 16 + (b1 - 0xF0)) * 80;
  if (k == 2) { hi[0] = lo[0] + 79; return 1; }
  lo[0] += pre[2] - 0x80; hi[0] = lo[0];
  return 1;
}
static void Add(const unsigned char* t, size_t l) {   // to the current word
  if (l > kWort - z.nc) l = kWort - z.nc;
  memcpy(z.cur + z.nc, t, l);
  z.nc += (unsigned)l;
}
static void Folge(unsigned char c) {  // the state behind byte c
  if (c == 0x0B) return;
  if (z.np || c >= 0x80) {
    z.pend[z.np++] = c;
    const int L = z.pend[0] <= 0xCF ? 1 : z.np < 2 ? 0 : z.pend[1] <= 0xCF ? 2 : 3;
    if (L && z.np == L) {
      size_t k = 0;
      const long r = dict_codes::Rang(z.pend, (size_t)L, &k);
      if (r >= 0 && (size_t)r < z.nw) {
        size_t l = 0;
        const unsigned char* t = Text((unsigned)r, &l);
        Add(t, l);
      }
      z.np = 0;
    }
    return;
  }
  if (c == 0x0A) {
    if (!z.nc) {
      z.hinter = true;
    } else if (z.nw <= 54857) {   // the current word is finished
      z.ab[z.nw++] = z.nt;
      const unsigned lc = z.nc < kTxt - z.nt ? z.nc : kTxt - z.nt;
      memcpy(z.txt + z.nt, z.cur, lc);
      z.nt += lc;
      z.nc = 0;
      z.hinter = false;
    }
    return;
  }
  Add(&c, 1);
}
// some code word beginning with pre[0..k) names a rank < i; with genau (exact)
// (inside an open code word, k >= 2: at most 80 ranks) one of them names a
// word T that keeps the order, S[0..ls) the rest of the previous word (equal
// so far): behind a mark T < S, in a run T >= S or T a proper prefix of S
static bool CodeOk(const unsigned char* pre, int k, unsigned long long i, bool genau,
                   const unsigned char* S, size_t ls) {
  unsigned lo[3], hi[3];
  const int n = Intervalle(pre, k, lo, hi);
  for (int q = 0; q < n; ++q) {
    if (lo[q] >= i) continue;
    if (!genau) return true;
    const unsigned e = hi[q] < i - 1 ? hi[q] : (unsigned)(i - 1);
    for (unsigned r = lo[q]; r <= e; ++r) {
      size_t lt = 0;
      const unsigned char* t = Text(r, &lt);
      const int c = memcmp(t, S, lt < ls ? lt : ls);
      if (z.hinter ? (c < 0 || (!c && lt < ls)) : c >= 0) return true;
    }
  }
  return false;
}
__attribute__((noinline)) static int Menge(const unsigned char* b, size_t m, unsigned char* v, int* art) {
  if (m < z.gez) z.gez = z.nt = z.nw = z.nc = 0, z.np = 0, z.hinter = false;   // a new slot 1
  for (; z.gez < m; ++z.gez) Folge(b[z.gez]);
  int n = 0;
  *art = 3;
  if (m && b[m - 1] == 0x0B) {
    v[n++] = 0x0B;
    return n;
  }
  const unsigned long long i = z.nw;             // index of the current word
  const bool regel = kRegel && i > dict_codes::kLaufAb;
  size_t lp = 0;
  const unsigned char* p = i ? Text((unsigned)(i - 1), &lp) : nullptr;
  const size_t k = z.nc;
  const bool eq = regel && k <= lp && (!k || !memcmp(z.cur, p, k));
  int lo = 0x61, hi = 0x7A, clo = 0x80, chi = 0xFF;
  bool lf = false, ende = false, genau = false;
  if (z.np) {
    lo = 1, hi = 0;                              // inside a code word: no letter
    chi = (z.np == 1 && z.pend[0] >= 0xF0) ? 0xFF : 0xCF;
    if (eq) genau = true, *art = 4;
  } else if (!m) {
  } else if (!k && b[m - 1] == 0x0A) {           // the start of a word
    if (z.hinter) {
      if (regel) hi = lp > 1 ? p[0] : p[0] - 1, *art = 4;
    } else {
      ende = true;
      lf = i > dict_codes::kLaufAb;           // a run mark
      if (regel) lo = p[0], *art = 4;
    }
  } else {
    // inside a word: no code word (the prefix rule); the line feed only
    // behind a letter
    clo = 1, chi = 0;
    lf = b[m - 1] < 0x80;
    if (eq && !z.hinter) {
      lf = false, *art = 4;
      if (k < lp) lo = p[k];
    } else if (eq && k < lp) {
      hi = k + 1 < lp ? p[k] : p[k] - 1, *art = 4;
    }
  }
  if (lf) v[n++] = 0x0A;
  if (ende) v[n++] = 0x0B;
  for (int c = lo; c <= hi; ++c) v[n++] = (unsigned char)c;
  unsigned char pre[3] = {z.pend[0], z.pend[1], 0};
  for (int c = clo; c <= chi; ++c) {
    pre[z.np] = (unsigned char)c;
    if (kRegel && !CodeOk(pre, z.np + 1, i, genau, p + k, lp - k)) continue;
    v[n++] = (unsigned char)c;
  }
  return n;
}
}  // namespace dict_runs
__attribute__((noinline)) static bool DictArticleSep() {
  unsigned char v[156];
  int n = 0;
  if (metablock::mk.TaNaechsterSlot() == 1) {
    n = dict_runs::Menge(nullptr, 0, v, &metablock::zw::art);
  } else if (metablock::mk.TaByte() && metablock::mk.TaSlot() == 1 &&
             !metablock::mk.TaSlotEndet()) {
    size_t m = 0;
    const unsigned char* b = metablock::mk.TaPuffer(1, &m);
    n = dict_runs::Menge(b, m, v, &metablock::zw::art);
  } else {
    return false;
  }
  return metablock::zw::Setze(v, n);
}

__attribute__((noinline)) static void TextStartByte(Predictor* p) {
  if (MetablockInert()) return;
  metablock::zw::an = false;
  if (!metablock::mk.Endet() && !metablock::mk.TaByte()) return;
  {
    bool jetzt = metablock::mk.TaSlotEndet() && metablock::mk.TaSlot() == 1;
    if (jetzt) DictArticleSlotEnd();
    if (jetzt) p->PpmdWaermen();
  }
  if (metablock::mk.TaSlotEndet() && metablock::mk.TaSlot() == 2) {
    p->WordlblBlockEnde();
  }
  const int s = metablock::mk.TaNaechsterSlot();
  if (s) TextStartHead(p, s);
  const int j = metablock::mk.TaZeileFolgt();
  if (j >= 0) {
    const unsigned char* q = nullptr;
    const int n = metablock::mk.TaOkb(j, &q);
    // the transformer gets the context as well - an author
    // prompt block like the slot head (fed, not predicted, masked in the dump)
    if (n) tf_author_prompt::Beginn(n);
    ObserveKnownBytes(p, q, n);
    if (n) tf_author_prompt::Ende();
  }
  // stage 2: a byte of a repetition line follows - its known bits
  metablock::zw::an = metablock::mk.TaZwSetze();
  if (!metablock::zw::an) metablock::zw::an = DictArticleSep();   // slot 1
  if (metablock::mk.TaFertig()) {
    // the transform grammar holds from the first byte of page 0 on (the slots
    // carry text outside it: the dictionary in plain text, the census texts)
    run_mode::gram.ZoneOeffnen(metablock::mk.Pos());
    p->MetalrBlockEnde();
  }
  (void)p;
}

ENC_ONLY static void TextStartSepEnc(Encoder& e, unsigned int c, unsigned long long pos) {
  unsigned int ctx = 1;
  const int a = metablock::zw::art; (void)a;
  for (int j = 7; j >= 0; --j) {
    const int bit = (int)((c >> j) & 1u);
    const int f = metablock::zw::line_buf[ctx & 255u];
    if (f < 0) {
      e.Encode(bit);
    } else {
      if (f != bit) {
        msg_stop(542);
        exit(3);
      }
      e.ObserveKnownBit(bit);
    }
    ctx += ctx + (unsigned int)bit;
  }
}
__attribute__((noinline)) static int TextStartSepDec(Decoder& d) {
  int byte = 1;
  while (byte < 256) {
    const int f = metablock::zw::line_buf[byte];
    int bit;
    if (f < 0) {
      bit = d.Decode();
    } else {
      bit = f;
      d.ObserveKnownBit(bit);
    }
    byte += byte + bit;
  }
  return byte;
}

// After every coded (Compress) or decoded (Decompress) stream byte c: the
// units and automata that read the stream, and the bytes they inject.  ONE
// function for both directions, so coder and decoder run the same code in
// the same order; noinline keeps it out of the two byte loops.
__attribute__((noinline)) static void AfterStreamByte(Predictor* p, unsigned char c) {
  if (!MetablockInert()) {
    metablock::mk.Byte(c);
    if (metablock::mk.MetaEndet()) {
      ppmd_meta::block_fertig = true;
    }
    if (metablock::mk.Endet()) {
      p->WordlblBlockEnde();
    }
  }
  // The byte is fully coded, so it is fixed on both sides.  If it closed a
  // title span, the alias rows of this article are pushed through the
  // transformer here, and through no other model: observed, never coded.
  // Beginn/Ende (begin/end) bracket them for it.  Mirrored in Decompress().
  alias_context::aut.Observe(c);
  {
    const unsigned char* blk = nullptr;
    const int n = alias_context::aut.TakeBlock(&blk);
    if (n) alias_prompt::Beginn(n);
    if (n) alias_title_subst::Beginn(alias_context::aut.PraefixLen());
    for (int k = 0; k < n; ++k) p->ObserveTfOnlyByte(blk[k]);
    alias_prompt::Ende();
    alias_title_subst::Ende();
  }
  if (!AuthorBlockInert()) author_block::aut.Observe(c);
  {
    const unsigned char* ab = nullptr;
    const int an = author_block::aut.TakeBlock(&ab);
    // The transformer must classify these bytes as a PROMPT block (feed 1
    // / predict 0), exactly as the alias prompt form does - mirrored, or
    // the archive is unreadable.
    if (an) tf_author_prompt::Beginn(an);
    ObserveKnownBytes(p, ab, an);
    tf_author_prompt::Ende();
  }
  CensusBlockByte(p, c);
  if (census_block::aut.ZeileEndet() && !metablock::mk.TaByte()) {
    p->MetalrBlockEnde();
    ppmd_meta::zeile_fertig = true;
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Observe(c);
    const unsigned char* tb = nullptr;
    const int tn = ts_progress::aut.TakeBlock(&tb);
    if (tn) timestamp_skip::Beginn();
    ObserveKnownBytes(p, tb, tn);
    timestamp_skip::Ende();
  }
  TextStartByte(p);
  (void)p; (void)c;
}

void Compress(unsigned long long input_bytes, FILE* is,
    FILE* os, unsigned long long* output_bytes, Predictor* p) {
  Encoder e(os, p);

  unsigned long long percent =
      1 + (input_bytes / (unsigned long long)progress_steps);
  unsigned long long prog_left = 1;
  ClearOutput();
  // the read block (BSS, one coder at a time per process)
  static char buffer[(size_t)io_buffer_kb * 1024];
  const size_t buffer_size = sizeof buffer;
  unsigned long long pos = 0;
  int sem_pb = 0;   // the previous stream byte (0 before the first)
  while (pos < input_bytes) {
    const size_t chunk = static_cast<size_t>(
        std::min<unsigned long long>(buffer_size, input_bytes - pos));
    const size_t got = fread(buffer, 1, chunk, is);
    if (got == 0) break;
    for (size_t i = 0; i < got; ++i) {
      unsigned char c = static_cast<unsigned char>(buffer[i]);
      p->SemByteStart(sem_pb);
      sem_pb = c;
      const int zw = p->Tf1ZwangByte();
      if (zw >= 0) {
        for (int j = 7; j >= 0; --j) e.ObserveKnownBit((zw >> j) & 1);
        if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)zw);
      } else
      if (metablock::zw::an) TextStartSepEnc(e, c, pos);
      else
      if (run_mode::gram.ByteActive(pos)) {
        unsigned int ctx = 1;
        for (int j = 7; j >= 0; --j) {
          const int bit = (c >> j) & 1;
          const int f = run_mode::gram.Forced(ctx);
          if (f < 0) {
            run_mode::gram.CountFree(ctx);
            e.Encode(bit);
          } else {
            run_mode::gram.CountKnown(ctx);
            if (f != bit) run_mode::gram.Violation(pos, ctx, f, bit);
            e.ObserveKnownBit(bit);
          }
          ctx += ctx + bit;
        }
        run_mode::gram.NextByte(c);
      } else
      {
      for (int j = 7; j >= 0; --j) {
        e.Encode((c>>j)&1);
      }
      }
      AfterStreamByte(p, c);   // the readers of the stream, mirrored in Decompress()
      if (--prog_left == 0) {
        prog_left = percent;
        double frac = 100.0 * pos / input_bytes; (void)frac;

        HUGEPAGE_TICK(pos);
        console_progress::Tick(pos, input_bytes);
      }
      ++pos;
    }
  }
  e.Flush();
  run_mode::gram.Report("compress");
  if (!MetablockInert()) {
    metablock::mk.Report("compress");
    census_block::aut.Report("compress");
    metablock::mk.ByteWacht("compress", pos);
  }
  redirect_table::tab.Schluss();
  alias_table::tab.Schluss();
  alias_context::aut.Finish();
  alias_context::aut.Report("compress");
  if (!AuthorBlockInert()) {
    author_block::aut.Finish();
    author_block::aut.Report("compress");
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Finish();
    ts_progress::aut.Report("compress");
  }
  *output_bytes = ftell(os);
}

void Decompress(unsigned long long output_length, FILE* is,
                FILE* os, Predictor* p) {
  Decoder d(is, p);
  unsigned long long percent = 1 + (output_length / 10000);
  unsigned long long prog_left = 1;
  ClearOutput();
  int sem_pb = 0;   // the previous stream byte (0 before the first)
  for(unsigned long long pos = 0; pos < output_length; ++pos) {
    int byte = 1;
    p->SemByteStart(sem_pb);
    const int zw = p->Tf1ZwangByte();
    if (zw >= 0) {
      for (int j = 7; j >= 0; --j) d.ObserveKnownBit((zw >> j) & 1);
      byte = 256 + zw;
      if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)zw);
    } else
    if (metablock::zw::an) byte = TextStartSepDec(d);
    else
    if (run_mode::gram.ByteActive(pos)) {
      while (byte < 256) {
        const int f = run_mode::gram.Forced((unsigned int)byte);
        int bit;
        if (f < 0) {
          run_mode::gram.CountFree((unsigned int)byte);
          bit = d.Decode();
        } else {
          run_mode::gram.CountKnown((unsigned int)byte);
          bit = f;
          d.ObserveKnownBit(bit);
        }
        byte += byte + bit;
      }
      run_mode::gram.NextByte((unsigned int)byte & 255);
    } else
    {
    while (byte < 256) {
      byte += byte + d.Decode();
    }
    }
    putc(byte, os);
    sem_pb = byte & 255;   // the stream byte for the next SemByteStart, as in Compress()
    AfterStreamByte(p, (unsigned char)(byte & 255));   // as in Compress()
    if (--prog_left == 0) {
      prog_left = percent;
      double frac = 100.0 * pos / output_length; (void)frac;
      HUGEPAGE_TICK(pos);
      console_progress::Tick(pos, output_length);
    }
  }
  run_mode::gram.Report("decompress");
  if (!MetablockInert()) {
    metablock::mk.Report("decompress");
    census_block::aut.Report("decompress");
    metablock::mk.ByteWacht("decompress", output_length);
  }
  redirect_table::tab.Schluss();
  alias_table::tab.Schluss();
  alias_context::aut.Finish();
  alias_context::aut.Report("decompress");
  if (!AuthorBlockInert()) {
    author_block::aut.Finish();
    author_block::aut.Report("decompress");
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Finish();
    ts_progress::aut.Report("decompress");
  }
}

bool Store(const std::string& input_path, const std::string& temp_path,
    const std::string& output_path, FILE* dictionary,
    unsigned long long* input_bytes, unsigned long long* output_bytes) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;
  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  WriteStorageHeader(data_out, dictionary != NULL);
  preprocessor::Encode(data_in, data_out, *input_bytes, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(data_in);
  fclose(data_out);
  return true;
}

bool RunCompression(bool enable_preprocess, const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* temp_out = fopen(temp_path.c_str(), "wb");
  if (!temp_out) return false;

  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);

  if (enable_preprocess) {
    preprocessor::Encode(data_in, temp_out, *input_bytes, dictionary);
  } else {
    preprocessor::NoPreprocess(data_in, temp_out, *input_bytes);
  }
  fclose(data_in);
  fclose(temp_out);

  if (census_block_active && !census_block::Einfuegen(temp_path, ".censblock", dictionary)) {
    msg_note(407);
    return false;
  }
  if (!metablock_form::Pack(temp_path)) {
    msg_note(171);
    return false;
  }

  if (!ip_pack::Pack(temp_path)) {
    msg_note(172);
    return false;
  }
  if (!MetablockInert() && !dict_article::SlotDatei()) {
    msg_note(547);
    return false;
  }
  if (!MetablockInert() && !rule_stages::DateiEnc(temp_path)) {
    msg_note(553);
    return false;
  }
  if (!text_start::Pack(temp_path, !MetablockInert(), !enable_preprocess)) {
    msg_note(536);
    return false;
  }

  console_progress::PhaseEnd();   // "preprocessing" ends here

  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;

  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  fseek(temp_in, 0L, SEEK_END);
  unsigned long long temp_bytes = ftell(temp_in);
  fseek(temp_in, 0L, SEEK_SET);

  std::vector<bool> vocab(256, false);
#if AUX_BUILD
  vocab_frozen::g_aux = true;
#endif
  vocab_frozen::FillVocab(&vocab);
  WriteHeaderVocfix(temp_bytes, dictionary != NULL, data_out);

  dict_article::strom = temp_bytes;
  dict_article::wrt = dictionary != NULL;
  if (!MetablockInert()) dict_article::Verbergen();
  Predictor p(vocab);
  HUGEPAGE_COLLAPSE("c-init");
  malloc_trim(0);
  console_progress::PhaseEnd();   // "pretraining ..." ends here
  Compress(temp_bytes, temp_in, data_out, output_bytes, &p);
  if (!MetablockInert()) dict_article::Pruefen();
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

bool RunDecompression(const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;

  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  std::vector<bool> vocab(256, false);
  bool dictionary_used;
  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);
  if (!dictionary_used && dictionary != NULL) return false;

  {
    dict_article::strom = *output_bytes;
    dict_article::wrt = dictionary_used;
    if (!MetablockInert()) dict_article::Verbergen();
    Predictor p(vocab);
    HUGEPAGE_COLLAPSE("d-init");

    FILE* temp_out = fopen(temp_path.c_str(), "wb");
    if (!temp_out) return false;

    console_progress::PhaseEnd();   // "pretraining ..." ends here
    Decompress(*output_bytes, data_in, temp_out, &p);
    // the inverse transforms of the main run take some minutes
    if (dictionary_used) console_progress::Phase("postprocessing");
    p.FreeFxcmMemory();
    p.FreeFxcmRest();
    fclose(data_in);
    fclose(temp_out);
  }
  malloc_trim(0);
  if (!text_start::Unpack(temp_path, !MetablockInert())) {
    msg_note(537);
    return false;
  }
  if (!MetablockInert()) {
    if (!dict_article::DateiAusSlot()) {
      msg_note(548);
      return false;
    }
    if (dict_article::wrt) {
      if (dictionary) fclose(dictionary);
      dictionary = fopen(".dict", "rb");
    }
  }
  if (!MetablockInert() && !rule_stages::DateiDec(temp_path)) {
    msg_note(554);
    return false;
  }

  if (!ip_pack::Unpack(temp_path)) {
    msg_note(173);
    return false;
  }

  if (!metablock_form::Unpack(temp_path)) {
    msg_note(174);
    return false;
  }
  if (census_block_active && !census_block::Entnehmen(temp_path, ".censblock", dictionary)) {
    msg_note(408);
    return false;
  }

  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  preprocessor::Decode(temp_in, data_out, dictionary);
  isbn_check::after_at.Put(-1, data_out);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

namespace ppmd_page { extern int hilfs; }
int main(int argc, char** argv) {
  console_progress::Begin();
  if (argc == 1 && no_args::IsCompressor()) {
    argc = 4;
    argv = (char**)no_args::kArgVector.argv;
  }
#if !AUX_BUILD
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] | 0x20) == 'd')   // -d or -D
#else
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] == 'c' || argv[1][1] == 'd'
      || argv[1][1] == 'C' || argv[1][1] == 'D'))
#endif
    ppmd_page::hilfs = 1;
  mallopt(M_MMAP_THRESHOLD, 1 << 20);
  if (PredictorLayoutTagFast() != ((unsigned long long)sizeof(Predictor) * 4ULL + (unsigned long long)(1) * 2ULL + (unsigned long long)(0))) {
    msg_note(175)

;
    fflush(stderr);
    return -1;
  }
#if !AUX_BUILD
  // The release form takes -e (compress), -d (decompress; a fifth argument, a
  // dictionary path, is accepted and not read) and -D (an embedded block of
  // the self-extraction).  Builds without this restriction, the build helper
  // among them, also take -c/-n/-s/-x and -C.
  if ((argc != 1) && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'd' && argv[1][1] != 'e'
      && argv[1][1] != 'D'
      ))) {
    return Help();
  }
#else
  if ((argc != 1) && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd' && argv[1][1] != 'x' && argv[1][1] != 's' &&
      argv[1][1] != 'n' && argv[1][1] != 'e'
      && argv[1][1] != 'C' && argv[1][1] != 'D'
      ))) {
    return Help();
  }
#endif
#if !AUX_BUILD
  if (argc > 1 && argv[1][1] == 'D') {
#else
  if (argc > 1 && (argv[1][1] == 'C' || argv[1][1] == 'D')) {
#endif
    transformer::SetBlobForm();
#if !AUX_BUILD
    argv[1][1] = 'd';
#else
    argv[1][1] = (argv[1][1] == 'C') ? 'c' : 'd';
#endif
  }
  srand((unsigned int)run_options::ParamI("seed", SEED));
  progress_steps = run_options::ParamI("progress_steps", 10000);
  if (progress_steps < 1) progress_steps = 10000;

#if !1 || AUX_BUILD
  clock_t start = clock();   // only for the end line of the build helper
#endif

  bool enable_preprocess = true;
  std::string input_path ;
  std::string output_path;
  FILE* dictionary = NULL;

  if (argc > 1)  {
#if !AUX_BUILD
    // the last two arguments; a dictionary argument of -d is not read, the
    // dictionary is slot 1 of the stream
    input_path = argv[argc - 2];
    output_path = argv[argc - 1];
#else
    input_path = argv[2];
    output_path = argv[3];
    if (argc == 5) {
      if (argv[1][1] == 'n') return Help();
      dictionary = fopen(argv[2], "rb");
      if (!dictionary) return Help();
      input_path = argv[3];
      output_path = argv[4];
    }
#endif
  }

  std::string temp_path = output_path + ".cmix.temp";
  const bool progress_encoding = argc > 1 && (argv[1][1] == 'c' || argv[1][1] == 'e');
  console_progress::Start(progress_encoding, output_path.c_str());

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argc == 1) {
    //Decompress enwik9
    // unpack a) header b) cmix dictionary, c) new order of articles, d) actual cmix binary
    census_block_active = true;
    if (selfextract_decomp() != 0) {
      return Help();
    }

    // run compression
    input_path = ".ready4cmix_decomp";
    output_path = ".input_decomp" ;
    dictionary = fopen(".dict", "rb");
    isbn_check::after_at.an = true;

    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes,
        nullptr
        )) {
      return Help();
    }

    split4Decomp();

    // apply phda9 preprocessor
    phda9_resto();
    if (!census_block::Setze(".main_decomp_restored", ".censblock")) {
      msg_note(409);
      return Help();
    }
    if (!comment_form::Dekodiere(".main_decomp_restored")) {
      msg_note(484);
      return Help();
    }

    // change the order of articles in the input
    sort();

    if (!cmtidx_inplace<0>(".main_decomp_restored_sorted", ".cmtidx_resto")) {
      msg_note(176);
      return Help();
    }

    // merge all input parts after preprocessing; the census inverse transform
    // follows as the last step - mirror image of compression, where it ran first.
    {
      static const char* const teile[] = {".intro_decomp", ".main_decomp_restored_sorted",
                                          ".coda_decomp", nullptr};
      if (!cat(".census_resto", teile)) return Help();
    }
    if (!census_file<0>(".census_resto", "enwik9_uncompressed")) {
      msg_note(177);
      return Help();
    }
    remove(".census_resto");

    selfx_clean::AfterDecompress();

    goto print_end_message;
  }

#if !(1 && !AUX_BUILD)   // the package form has no -c
  if (argv[1][1] == 'c') {
      if (argv[1][1] == 'c' && argc == 5) {
        if (!provide_dict_file(argv[2])) return Help();
      }
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes)) {
      return Help();
    }
  } else
#endif
  if (argv[1][1] == 'e') {
#define ENC_FAIL return Help()

    if (selfextract_comp() != 0) {
      ENC_FAIL;
    }

    {
      console_progress::Phase("dictionary");
      const bool derived = dict_derive::BuildDictFromInput(
          input_path.c_str(), ".hints", ".dict");
      console_progress::PhaseEnd();
      if (!derived) {
        msg_note(178);
        ENC_FAIL;
      }
      remove(".hints");
      remove(".hints.comp");
    }

    console_progress::Phase("preprocessing");
    if (!census_file_enc(input_path.c_str(), ".census_prepr")) {
      msg_note(181);
      ENC_FAIL;
    }
    split4Comp(".census_prepr");
    remove(".census_prepr");
    if (!cmtidx_inplace_enc(".main", ".cmtidx_main")) {
      msg_note(182);
      ENC_FAIL;
    }

    // change the order of articles in the input
    reorder();
    if (!comment_form::Kodiere(".main_reordered")) {
      msg_note(483);
      ENC_FAIL;
    }
    if (!census_block::Zerlege(".main_reordered", ".censblock")) {
      msg_note(406);
      ENC_FAIL;
    }
    census_block_active = true;

    // apply phda9 preprocessor
    phda9_prepr();

    // merge all input parts after preprocessing
    {
      static const char* const teile[] = {".main_phda9prepr", ".intro", ".coda", nullptr};
      if (!cat(".ready4cmix", teile)) ENC_FAIL;
    }
    isbn_check::Pack();

    // run compression
    input_path = ".ready4cmix";
    dictionary = fopen(".dict", "rb");
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes,
        nullptr
        )) {
      ENC_FAIL;
    }

    // construct a selfextracting decompressor binary; in the release
    // configuration
    //   archive9 = [decoder][container][payload][trailer]
    // (the dictionary travels as slot 1 of the stream; a build without the
    // dictionary article puts .dict.comp behind the decoder, one without the
    // embedded container leaves the container out).
    // It is written under the temporary name first: the payload may be the
    // file archive9 itself (the jury form writes its output there).
    {
      ArchivKopf header;
      header.weights_size = (int)getFileSize("weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5");
      header.decomp_input_size = (int)getFileSize(output_path.c_str());
      const char* const teile[] = {".decomp_bin",
                                   "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5",
                                   output_path.c_str(), nullptr};
      if (!cat(temp_path.c_str(), teile, &header, sizeof(header)) ||
          rename(temp_path.c_str(), "archive9") != 0)
        ENC_FAIL;
    }
    // make the decompressor binary executable
    chmod("archive9", 0777);

    selfx_clean::AfterCompress();

#undef ENC_FAIL
  }
  else
  {
#if !(1 && !AUX_BUILD)
    if (argc == 5) {
      if (!provide_dict_file(argv[2])) return Help();
    }
#endif
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
  }

print_end_message:
#if !AUX_BUILD
  {
    const double progress_s = console_progress::End(progress_encoding);
    printf("%lld bytes -> %lld bytes in %1.2f s.\n", input_bytes, output_bytes,
           progress_s);
  }
#else
  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n",
      input_bytes, output_bytes,
      ((double)clock() - start) / CLOCKS_PER_SEC);
#endif

  return 0;
}
