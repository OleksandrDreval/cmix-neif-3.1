# Changes in `cmix-neif-3.1` (against `cmix-neif-2.0`)

SPDX-License-Identifier: GPL-3.0-only

* **Author of `cmix-neif-3.1` modifications:** **Oleksandr Dreval** (<https://github.com/OleksandrDreval>)
* **Repository:** <https://github.com/OleksandrDreval/cmix-neif-3.1>
* **Copyright & License:** Copyright (C) 2026 Oleksandr Dreval — distributed under **GNU General Public License v3.0 only (`GPL-3.0-only`)**, an official OSI-Approved Open Source License (<https://opensource.org/license/gpl-3-0>) and FSF Free Software License (GPL-3.0 section 5a modification notice; see `LICENSE` and `AUTHORS`):

1. **Packer, stage-1 bootstrap CM, single-pass forward `cm_compress_k` & self-extracting stub (`tools_src/packer/stub.c`, `tools_src/packer/boot_models.h`, `tools_src/packer/cm_core.h`, `tools_src/packer/pack_tool.c`, `tools_src/packer/build_packer.sh`, `build_and_construct_comp.sh`)**: moved the stage-2 warm-up `memcpy` and `x86t_range` transform ahead of the first `syscall` in `s2_main()` so user-space syscall interception (`systrap`) cannot corrupt RWX `s2_base` bytes before warm-up; re-tuned stage-1 bootstrap context-mixer parameters (`BOOT_EPS = 4`, `BOOT_WT_LIST` in `boot_models.h`); upgraded host `cmcut_tool` to `-O3 -march=native` and introduced `cm_compress_k()` (`K = a.outn + 4` at `i == nd`), computing both the full compressor stream and the exact truncated decompressor stream length `K` in a single forward `cm_ac_encode` pass without invoking `cm_decompress`, `boot_decompress`, or any unpacker (`9.12 GiB` -> `1.51 GiB` peak packer RSS, `~39.0 s` packer time, `4.06x` faster than Iteration 2).
2. **Multi-model probability-mixed & covariance-compensated `.c5` transformer weight container (`weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5`, `src/tf1/weights_io_compressed.cpp`, `src/tf1/weights_io.h`, `src/transformer.h`, `training/tf/3_train/export/np2/tf1_np_codec.cpp`)**: added Sign-Magnitude coding for `ENC_INT4W`, per-canonical-name (`ci`) probability tables, 16-bucket node-aware width context `b` (blending `w_mag` and `w_nd`), 5-way integer probability mixing (`>> 5`, `CNT_LIMIT_INT4W = 3071`) at each `ENC_INT4W` binary tree node with per-node weights combining `p_tree`, 2D spatial context `p_2d` (left, above, RoPE parity `k & 1`), 3D spatial context `p_3d` (left, left-2, upper-left + upper-right diagonal, above), Bayesian row-column node posterior `p_rc`, and leaf-node Bayesian row-column QAT off-grid `{1, 3, 5}` rate `p_grid` on `(p_tree + p_rc) >> 1`; per-row QAT-grid flag `gbit` (`zr_ctx[kNamenN * 4]`) skipping the 3rd magnitude tree level (`bit_k = 0`) on `{0, ±2, ±4, ±7}` rows (`if (gbit) node = (node << 1) | uint32_t(node == 7);`); 4-way integer probability mixing (`>> 5`) for `ENC_INT4W` signs combining `sign_int4w` (`14/32`), `sign_2d` (`6/32`), `sign_rc` (`4/32`), and continuous magnitude-weighted Bayesian row-column positive-dipole posterior `p_src` (`8/32`, `dr_s = 192, dc_s = 320`); row-$L_2$-energy-bucket (`rb in {0..3}`) conditioned `ENC_BF16C` low-byte shift-tree mixing (`6/16` per-`(ci, rb)` + `10/16` per-`ci`) and column-conditioned (`jb in {0..4}`) `ENC_F32HC` high-byte (`prev_col_hi[jb]`) and low-byte (`10/16` per-`(ci, jb)` + `3/16` per-`ci` + `3/16` shared) shift-tree mixing; header metadata byte-role context mixing (`cm_role[8 * 256]`), adaptive order-2/1/0 `cname` character tree mixing, and per-`ci` `prev_z`-conditioned zero-row `zr_ctx[kNamenN * 4]` mixing (losslessly shrinking `.c5` to `3,728,114 B`, `-5,011 B` vs Iter 3); plus fast-weight-preserving coupled exact-$L_2^2$-norm feature-Gram `mlp.up` (`blocks.6..11.mlp.up.weight.q` with Lehmer-order-preserving prefix snapping) and closed-form analytical $\text{ReLU}^2$ cross-covariance-compensated `mlp.down` (`blocks.4..15.mlp.down.weight.q`, $g_{\text{init}} = (C_{\hat{h},\hat{h}} - C_{\hat{h},h}) w_r$) snapping onto the true 4-level QAT grid `{0, ±2, ±4, ±7}` (`tf1_np_codec qat-snap`, `min_blk = 4, up_min_blk = 6, up_max_blk = 12`, preserving 100 % of `npkey` Lehmer digits `K = 20,277 B`) — shrinking the container from `3,744,629 B` to **`3,512,771 B`** (`-231,858 B` per copy / **`-463,716 B` in `S`** vs `cmix-neif-2.0`; `-222,671 B` per copy / `-445,342 B` in `S` vs Iter 3; FNV-1a-64 `0x844f1f2bf695ec2bULL`, SHA-256 `cda620662f617c81ceaba2559ea79db7acd179d12399e451c516fb5a126ff2fe`).
3. **BITLSTM compact 4-bit exponent+nibble weight packing, active-cell `H` fast-path & dead-code pruning (`src/bit_head_weights.h`, `src/bit_head.cpp`, `src/mixer_precond.cpp`, `src/mixer/sse.cpp`, `src/glibc_compat.cpp`, `src/models/match.cpp`)**: rounded slot-1 (`i in [2561, 13570)`) `bf16` mantissas from 4 bits to 3 bits (round-to-nearest-even) and packed each weight into a single byte `(e_code << 4) | s` in `kBlobW[24669]` (`e_code = e - 116` for `e in [116, 130]` covering `99.63%` of all `24,579` weights, and `e_code = 15` with a 90-byte exception tail `kBlobW[24579..24668]` indexed by `kSlotExcOff[3] = {0, 4, 55}`), cutting `kBlobW` `.rodata` in half from `49,158 B` to `24,669 B` (`-24,489 B` raw `.rodata`, `-49.8%`, `-18,640 B` stripped ELF `Nf`/`Nd`, and `-57 B` after stage-2 `cm_compress`); pruned all dead online-training functions and 39 unused pointer/scalar fields from `struct Head` and `Head::Free()` (`g_head` in `.bss` shrunk to `264 B`); restricted `Head::Forward()` to active cells `H = kSlotH[slot]` (`H = 16` via `Dot16Preloaded` (`Hsum4`) for slot 0 across all 30.99M metadata/census bits, cutting slot-0 `BITLSTM` compute by `~2.7x`, and `H = 32` via `Dot32Preloaded` (`Hsum8`) for slots 1 and 2); transposed `hidP[cc * nhid + r]` to vectorize `Adjust()` in non-FMA AVX2 with `0 ULP` difference against scalar IEEE-754 order; vectorized `mixer_precond.cpp` column scaling in AVX2; removed dead functions/shims (`sse.cpp`, `glibc_compat.cpp`), replaced `history_.at()` with `history_[]` in `match.cpp`, and annotated cold lifecycle routines with `COLD_SIZE_ATTR`.
4. **PPMd page pool, SEMCACHE & `word_label` lifecycle & allocators (`src/models/ppmd_page.h`, `src/models/ppmd_page.cpp`, `src/models/ppmd.cpp`, `src/semantic_cache.h`, `src/predictor.cpp`, `src/word_label.cpp`, `src/runner.cpp`, `src/arena_alloc.h`, `src/tf1/opt/arena_build.cpp`)**: reduced `DYN_KLEMME_KIB` to `3.5 GiB`, max PPMd pool to `1 GiB` (`64 MiB` helper), and initial `StartSubAllocator` buffer to `256 MiB` (`64 MiB` helper); fixed `zonen_textrahmen(N)` rounding granularity from `16384` to `1024` frames (`4 MiB`) and removed the phantom `+2.12 GiB` `ppmd_page::raum(kSemTotal)` pre-charge so `PPMd` retains its full `1 GiB` pool under the `3.5 GiB` clamp; fixed the `69 MiB` `wordlbl_` memory leak in `Predictor::FreeFxcmRest()` and removed `PAGE_REGISTRY_ADD` on transient `word_label` / `SemTable` buffers to eliminate dangling pointers in `pageregistry::T()` and premature `MADV_COLLAPSE` on 50.3 MiB of untouched zero pages; kept exact `SEMCACHE` geometry (`kNLog = 24, kB = 18`) in a single demand-paged `MAP_ANONYMOUS | MAP_NORESERVE` mapping unmapped in `FreeFxcmRest()`, switched `arena::Bump::NeuerBlock()` to demand-paged 4 KiB `MAP_NORESERVE` blocks, removed redundant `memset` calls on kernel-zeroed mappings, and unrolled `SemDotPm1<N>` 4 rows at a time in AVX2 — lowering full-run peak RAM by `2.86x` (`9.77 GiB` -> `~3.34 GiB`).
5. **Compiler warning cleanups, build flags & attribution (`makefile`, `src/`, `AUTHORS`)**: resolved `-Wdangling-else`, `-Wbitwise-instead-of-logical`, `-Wparentheses`, `-Wconstant-logical-operand`, `-Wunused-private-field`, `-Wunused-function`, and `-Wunused-variable` diagnostics across `src/`, added `-Wno-backend-plugin` to `prof_use` and `cmix_aux` in `makefile`, added `SPDX-License-Identifier: GPL-3.0-only` modification headers to touched files, and documented all four optimization iterations in `en-optimization_report_it_1.md`, `uk-optimization_report_it_1.md`, `en-optimization_report_it_2.md`, `uk-optimization_report_it_2.md`, `en-optimization_report_it_3.md`, `uk-optimization_report_it_3.md`, `en-optimization_report_it_4.md`, `uk-optimization_report_it_4.md`, and `AUTHORS`.

## Verified `cmix-neif-3.1` summary against `cmix-neif-2.0`

| metric / component | `cmix-neif-2.0` | `cmix-neif-3.1` (Iter 4) | delta |
|---|---:|---:|---:|
| `.c5` weight container (per file / in `S`) | 3,744,629 B / 7,489,258 B | **3,512,771 B** / **7,025,542 B** | **-231,858 B** / **-463,716 B** (`-222,671 B` / `-445,342 B` vs Iter 3) |
| `BITLSTM` `kBlobW` `.rodata` & stripped ELF (`Nf` / `Nd`) | 49,158 B (`619,918` / `538,880` B) | **24,669 B** (`601,822` / `520,768` B) | **-24,489 B `kBlobW` (-49.8 %)** / **-18,112 B ELF** (`-18,640 B` vs Iter 3) |
| Packed compressor + decompressor (`cmix_orig + .decomp_bin`) | 395,323 B (211,438 + 183,885) | **397,117 B** (212,328 + 184,789) | `+1,794 B in S` (includes multi-model `.c5` decoder yielding `-463,716 B` in `.c5`; `-12,148 B` vs Iter 3) |
| Shipped `comp9` / projected `S = comp9 + archive9` | 4,042,564 B / 93,388,956 B | **3,811,596 B** / **92,927,013 B** | **-230,968 B** on `comp9` / **-461,943 B in `S`** (`-460,872 B` vs Iter 3) |
| Peak resident set on `enwik9` | 9.77 GiB | **~3.34 GiB** | **-6.20 GiB (2.86x lower)** |
| Packer build peak RSS & single-pass time (`build_packer.sh`) | 9.12 GiB / >300 s (`systrap` trap) | **1.51 GiB** / **~38.0 s** (`204.0 s` full `build.sh`) | **6.05x lower RAM**, `4.06x` faster vs Iter 2 |
| Single-core coding / decoding speed (slices) | 1.00x | **+10.8 % to +11.2 %** (`~2.7x` faster `BITLSTM` slot 0) | 100 % bit-identical round-trip |

---

# Upstream changes in `cmix-neif-2.0` (against `cmix-neif-1.0`, Marcus Wölk)

The reference point is `cmix-neif-1.0`, the previous version of this entry,
submitted on 5 October 2026 and published at
<https://github.com/marwoelk/cmix-neif-1.0>. Its own changes against the fork
base `cmix-lex` by Ibrahim Marcouch, derived from `fx2-cmix` (Kaido Orav, Byron
Knoll), are listed in its `CHANGES.md`. This version keeps the architecture of
`cmix-neif-1.0` and changes it in the eight groups below.

Besides the groups below, the code carries numerous small changes (renamings,
simplifications, error handling and console messages, removal of unused code,
and small run-time and memory improvements) that are not listed one by one;
group 8 gives the size of the program with all of them.

This file is the modification notice required by section 5a of the GNU
General Public License for the changes against `cmix-neif-1.0`. Sections 3 and
4 of `README.md` carry the same material with the measurements, the individual
items and the reasons; `NOTICE.md` names the borrowed parts and their authors.

## 1. SEMCACHE

A semantic cache over the states of the transformer. Where the transformer
gives its most probable next byte a probability below 0.99, a key of 128
values taken from the input of the MLP of its last layer is looked up in a
memory of about 16.8 M earlier states, indexed by 16 hash tables; the bytes
that followed the 32 most similar states, together with followed neighbours of
earlier lookups, give a distribution of the next byte. It feeds the mixers,
four inputs of BITLSTM and an end stage in front of the coder. The memory is
filled from coded bytes only and takes about 2.2 GB, which the page pool of
the PPMd model gives up.

## 2. Census transform

The machine-written census pages are split into the parts of their template,
their values and the text that deviates. Template choices and values move into
a census metadata block behind the metadata block, where a number model codes
each value against an estimate of an online ridge regression per kind of
place; the page keeps only the deviating text and the names. The transformer
receives what the decompressor knows from the block as fed rows. The census
pages take 21.7 MB of the coder stream instead of 79.1 MB, the block 9.3 MB.

## 3. Transformer

* The architecture and the training recipe of Vladimir Malinovskii's
  `cmix-neif-beehive`: attention residuals, one key and value head shared by
  all query heads, value embeddings in the attention layers, an MLP width of
  1024, a head that mixes the logits of the PPMd prior into the output, weight
  rows of 2, 3 or 4 bits and quantization-aware training with a size term as
  the cool-down. The new parts
  of its inference engine are ported. The weights start from those of
  `cmix-neif-beehive` and were trained further on the coder stream of this
  version, at the attention window of 2048 of `cmix-neif-1.0`
  (`cmix-neif-beehive`: 1536).
* A fast-weight head after Ryan Landay's `ex-girlfriend 1.0`: a second
  distribution of the transformer from fast weights on the MLP blocks 12 to
  15, learned online while coding, as a further expert of the mixer.
* Container formats with entropy coding of the weights and with the order of
  interchangeable groups of neurons coded into the choice of their permutation.
* The rows fed after digit runs and the special paths for ISBN check digits
  are removed; the transformer reads the preprocessed stream as it is.
* Rows whose output nobody reads are computed as blocks, the depth mixture runs
  as an online softmax, and the dense kernels prefetch further ahead.

## 4. Stream form: text start and reversible text rules

* Behind the two metadata blocks the text begins with three articles. The
  first holds the dictionary in code form; the decompressor takes its
  dictionary from it, so the archive carries no separate dictionary block. The
  second holds the names of the contributors, which part B of the metadata
  block replaces by a mark (new or repeated name), with the day of each
  revision and the kind of mark fed as known bytes; anonymous addresses and a
  few names stay in part B. The third is a pseudo-article with the texts of the census template,
  each once, and the words of the states and kinds of place. The PPMd model is warmed up with the dictionary
  codes in front of the first article and with the word stems at its end.
* Three reversible rules on the page text: an indefinite article is written as
  the code of `a` where a rule on the following word gives the right form, and
  as the code of `an` where it does not; repetitions of the page
  title in interlanguage links, in the first bold text and in the sort keys of
  category links become a short mark; a link whose shown text follows from its
  target keeps only the target and a mark.
* A correct ISBN-10 check digit leaves the stream and is recomputed by the
  decompressor.
* Behind the redirect block, an edit comment that repeats an earlier comment of
  the same contributor becomes the number of that comment.
* Computed values (unit conversions, pairs of infobox fields, years) are fed to
  the transformer as blocks.
* The pieces of the link context are chosen by fixed rules and carry the title
  of their source page; a block mark stands in front of the title.
* Partial-word codes of the dictionary transform only as prefixes from 7
  letters on.
* After Ryan Landay's `ex-girlfriend 1.0`, the revision ids below 17,000,000
  are coded as remainders against an interpolation over the page id, and the
  timestamp forecast follows the slope of the last records above the fixed rank.

## 5. BITLSTM

Three weight sets, one with 16 cells for the metadata blocks, where BITLSTM
was off before, and two with 32 cells instead of 16 for the text up to the
calendar block and for the rest, each trained offline on a graphics card and
none of them learning while coding; the program has no backward pass. New
inputs from SEMCACHE and from the transformer (the fast-weight logit and the
path state and confidence of its most probable byte); the one-hot of the bit
position takes the true position of the bit in its byte.

## 6. Mixer, context models and PPMd

* The small online byte LSTM, its additional inputs and its PPMd prior are
  removed.
* A Bayesian mixture of four experts with a fixed share as an additional term
  of the final mixer, and a mixture of the SSE output with the mixer output.
* 24 additional rows of the final mixer by the state of the most probable byte
  of the transformer; the confidence of the transformer as context of one
  layer-0 mixer, adapted from `cmix-neif-beehive`.
* New mixer learning rates, a longer schedule, normalised inputs of the
  layer-0 mixers, a lower floor for the transformer probability and a lower
  input clamp; a floor stage with a 32-bit probability behind BITLSTM;
  the threshold of adaptive depth lowered from 8 to 7.
* The grammar automaton marks seven further kinds of fixed bits.
* The PPMd model restarts at the end of the metadata block and at the end of
  the census metadata block, and its paging layer with it; the page pool starts
  at 1 GiB; the tables of SEMCACHE are made resident at the first byte of the
  stream.

## 7. Speed

The time saved by the census transform, by the removal of the small byte LSTM,
by the faster kernels of SEMCACHE and by the block form of unread transformer
rows pays for SEMCACHE, the larger transformer and the fast-weight head.

## 8. Packaging

* A new packer for the self-extracting stub. Its first stage is a small context
  mixer over byte masks of the last bytes in place of an order-1 coder. Its
  second stage adds per context the run and byte history after paq8px, a mixer
  and a mixer layer selected by the role of the byte and the opcodes, and a
  model of jump targets. On the same program it is 18,665 B of `S` smaller than
  the packer of `cmix-neif-1.0`.
* The packed program, the compressor part of `comp9` plus the decompressor at
  the head of `archive9`, is 395,323 B against 348,267 B in `cmix-neif-1.0`,
  with all additions of this version and numerous small changes that make the
  code smaller (among them messages that print a code `E<n>` listed in
  `ERRORS.md`, an order of the functions in the link, and the removal of unused
  code, of constant slots of the text model and of the fallback path of the
  PPMd model on a full model area).
* The dictionary hints and the article-order residual are coded by a small
  context-mixing coder of their own.
* `comp9` started without arguments compresses `enwik9` into `archive9`, so
  `comp9.args` is empty.

## Result (`cmix-neif-2.0` baseline against `cmix-neif-1.0`)

| | bytes (`cmix-neif-2.0` baseline) |
|---|---:|
| `archive9` of `cmix-neif-2.0`, self-extracting | 89,346,392 |
| `comp9` of `cmix-neif-2.0`, the compressor | 4,042,564 |
| `comp9.args` | 0 |
| **`S` = `comp9` + `archive9` + `comp9.args` (`cmix-neif-2.0`)** | **93,388,956** *(before `-340,197 B` weight/code/packer savings in `cmix-neif-3.1`)* |
| for comparison, `cmix-neif-1.0` | 95,914,051 |

## What was adopted from other entries

Named here so that it is visible without reading the register; `NOTICE.md`
lists the items with licences.

New in this version:

* from **Vladimir Malinovskii** (`cmix-neif-beehive`): the architecture and
  the training recipe of the transformer, the new parts of its inference
  engine, the prefetch depth of the dense kernels, the confidence of the
  transformer as mixer context, the training scripts on which the trainer of
  this entry is built, and the weights from which the training of the shipped
  weights started;
* from **Ryan Landay** (`ex-girlfriend 1.0`): the fast-weight head, the
  interpolation of revision ids over the page id and the timestamp forecast
  from the slope of the last records, adapted from his code;
* from **paq8px** v217 (the run map by **Zoltán Gotthardt**): the run and byte
  history per context in the packer, written anew;
* indirect contexts over the instruction in the packer, like the indirect
  context model of **Matt Mahoney's** paq8f, written anew;
* from **Matt Mahoney's** `lpaq1`: the squash and stretch functions, the
  arithmetic coder and the counter rate of the coder of the hints and the order
  residual, and the squash table in the sign rule of the census region;
* from fdlibm (**Sun Microsystems**), with the pipelined evaluation of
  **Naohiko Shimizu**, through glibc 2.39: `log1p`;
* the generator xorshift64* (**Sebastiano Vigna**, on **George Marsaglia**'s
  xorshift, with a multiplier from the tables of **Pierre L'Ecuyer**) for the
  fixed random projections of SEMCACHE, and its multiplier in the context hash
  of the packer;
* the golden-ratio constant of multiplicative hashing as described by **D. E.
  Knuth**, and in new places the multiplier of the MurmurHash3 finalizer
  (**Austin Appleby**) and the FNV primes (**Glenn Fowler, Landon Curt Noll,
  Kiem-Phong Vo**), as hash constants;
* from the literature: attention residuals (**Kimi Team**), evaluated as an
  online softmax (**M. Milakov, N. Gimelshein**), multi-query
  attention (**Noam Shazeer**), value embeddings (**Braden Koszarsky**,
  Modded-NanoGPT), the factorial number system of a permutation
  (**C.-A. Laisant**) and the bits-back argument (**G. E. Hinton, D. van
  Camp**) in the container of the weights, nearest-neighbour language models
  (**Khandelwal et al.**), the continuous cache and its unbounded form (**Grave
  et al.**), SimHash (**M. Charikar**), multi-probe hashing (**Q. Lv et al.**),
  AdaGrad (**J. Duchi, E. Hazan, Y. Singer**), the fixed share (**M. Herbster,
  M. K. Warmuth**), ridge regression (**A. E. Hoerl, R. W. Kennard**) in the
  number coder of the census transform, and context models over byte masks in
  a small stub as in Crinkler (**Aske Simon Christensen, Rune L. H. Stubbe**),
  like in oneKpaq
  (**Teemu Suutari**).

Kept from `cmix-neif-1.0`: the transformer inference engine of **Vladimir
Ivanov** (`fx2-cmix-transformer`); BITLSTM and the zero-state masking of
**David Freelan** (`cmix-obias`); adaptive depth, indel recovery, the
end-to-end mixer gradient, the word-label model and four further items of
**James Byrne** (`zmix 1.0`); the second match channel and the last step of
the learning-rate schedule of **Halvor Yttredal** (`fx-deepmix`); the items of
**Rohit Dwivedula**, **Layth Lakhani**, **Seth Lupo** and **mpais**; and the
earlier work listed in the `CHANGES.md` of `cmix-neif-1.0`.

With the small byte LSTM its PPMd prior (**David Freelan**), its
block-diagonal recurrence (**Xeloraa**), its eight-wide `tanh` with the Cephes
polynomial (**Stephen L. Moshier**) and the xxHash32 steps of its pair and
triple hashes (**Yann Collet**) were removed and are no longer used. The feed of its prediction into
the text model (the wiring of **Kaido Orav** in the form of **Halvor
Yttredal's** `fx-deepmix`) is still in the code and now carries a constant.
