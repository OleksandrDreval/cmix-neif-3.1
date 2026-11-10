# Changes in `cmix-neif-3.1` (against `cmix-neif-2.0`)

SPDX-License-Identifier: GPL-3.0-only

* **Author of `cmix-neif-3.1` modifications:** **Oleksandr Dreval** (<https://github.com/OleksandrDreval>)
* **Repository:** <https://github.com/OleksandrDreval/cmix-neif-3.1>
* **Copyright & License:** Copyright (C) 2026 Oleksandr Dreval — distributed under **GNU General Public License v3.0 only (`GPL-3.0-only`)**, an official OSI-Approved Open Source License (<https://opensource.org/license/gpl-3-0>) and FSF Free Software License (GPL-3.0 section 5a modification notice; see `LICENSE` and `AUTHORS`):

1. **Packer, stage-1 bootstrap CM, single-pass forward `cm_compress_k` & self-extracting stub (`tools_src/packer/stub.c`, `tools_src/packer/boot_models.h`, `tools_src/packer/cm_core.h`, `tools_src/packer/pack_tool.c`, `tools_src/packer/build_packer.sh`, `build_and_construct_comp.sh`)**: moved the stage-2 warm-up `memcpy` and `x86t_range` transform ahead of the first `syscall` in `s2_main()` so user-space syscall interception (`systrap`) cannot corrupt RWX `s2_base` bytes before warm-up; re-tuned stage-1 bootstrap context-mixer parameters (`BOOT_EPS = 4`, `BOOT_WT_LIST` in `boot_models.h`); upgraded host `cmcut_tool` to `-O3 -march=native` and introduced `cm_compress_k()` (`K = a.outn + 4` at `i == nd`), computing both the full compressor stream and the exact truncated decompressor stream length `K` in a single forward `cm_ac_encode` pass without invoking `cm_decompress`, `boot_decompress`, or any unpacker (`9.12 GiB` -> `1.51 GiB` peak packer RSS, `~39.0 s` packer time, `4.06x` faster than Iteration 2).
2. **Multi-model split-Dirichlet probability-mixed, continuous-target $\alpha$-rescaled multi-candidate Lagrangian Rate-Distortion & covariance-compensated `.c5` transformer weight container (`weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5`, `src/tf1/weights_io_compressed.cpp`, `src/tf1/weights_io.h`, `src/transformer.h`, `training/tf/3_train/export/np2/tf1_np_codec.cpp`)**: added Sign-Magnitude coding for `ENC_INT4W`, per-canonical-name (`ci`) probability tables, 16-bucket node-aware equal-weight density context `b = (w_mag + w_nd) >> 1`, split-Dirichlet priors (`prr = 32, pcc = 16` for marginal scale, `dr1 = 128, dc1 = 8` for bucket-index density `w_nd`, and `dr2 = 64, dc2 = 16` for posterior `p_rc`), and 4-way integer probability mixing (`>> 5`, `CNT_LIMIT_INT4W = 3071`) at each `ENC_INT4W` binary tree node with per-node weights (`wt = 22 - kW2d[node]`, `kW2d = {0, 3, 2, 2}`, `1/32` 3D, `9/32` `p_rc`) combining `p_tree`, 2D spatial context `p_2d` (left, above, RoPE parity `k & 1`), 3D spatial context `p_3d` (left, left-2, upper-left + upper-right diagonal, above), and Bayesian row-column node posterior `p_rc`; achieved **100 % `{0, ±2, ±4, ±7}` QAT-grid coverage across all 39,046 non-zero rows of all 154 `ENC_INT4W` tensors in Stream A (`g0_rows = 0`)**, eliminating per-row `gbit` flags (`zr_ctx` shrunk from `kNamenN * 4` to `kNamenN * 2`), shrinking `cint4w`, `c2d`, and `c3d` tree strides from `8` to `4`, replacing heap-allocated `last_zrow` with inline `last_zrow[1024]`, and shrinking `cname[65536]` to 14-bit ASCII `cname[16384]` (saving `~792 KiB` in `ModelleId` and eliminating 49,152 `free()` calls and 154 `calloc`/`free` pairs); 4-way integer probability mixing (`>> 5`) for `ENC_INT4W` signs combining `sign_int4w` (`12/32`), `sign_2d` (`6/32`), `sign_rc` (`4/32`), and continuous magnitude-weighted Bayesian row-column positive-dipole posterior `p_src` (`10/32`, `sdr = 256, sdc = 256`); pre-snap 4-bit mantissa quantization (`(val + 8) & 0xFFF0`) and unconditional 4-bit `lo`-nibble shift-tree coding (`nbits = 4`, `for (int bk = 7; bk >= 4; bk--)`) across **all** Stream A `ENC_BF16C` (`*.weight.scale`) and `ENC_F32HC` tensors (`dt == DT_F32`, including 36 KDA head groups `key_dims.h{0,1,2}` of `query_convolution.weight` via collision-free 4-bit `lo` rounding + Lehmer re-permutation), combined with row-$L_2$-energy-bucket (`rb in {0..3}`) `ENC_BF16C` and column-conditioned (`jb in {0..4}`) `ENC_F32HC` high/low mixing; header metadata byte-role context mixing (`cm_role[8 * 256]`), adaptive order-2/1/0 `cname` character tree mixing, and per-`ci` `prev_z`-conditioned zero-row mixing; plus continuous-target $\alpha$-rescaled (`\alpha_{\text{up}} = 1.24, \alpha_{\text{dn}} = 1.19`) coupled exact-$L_2^2$-norm feature-Gram `mlp.up` (`blocks.0..15.mlp.up.weight.q` with full 16-block Lehmer re-permutation), closed-form analytical $\text{ReLU}^2$ cross-covariance-compensated `mlp.down` (`blocks.0..15.mlp.down.weight.q` with exact $s_{\text{up}, j}^2$ $\text{ReLU}^2$ scaling and $\alpha_{\text{dn}}^{-1}$ cross-covariance target matching), multi-candidate (`{0, ±2s, ±4s, ±7s}`) Lagrangian Rate-Distortion / SparseGPT coordinate descent (`c247` + MDL rate penalty $R(w) \in \{0.0, 2.1, 3.4, 4.0\}\text{ bits}$, $\lambda_{\text{up}} = 1.17$, $\lambda_{\text{dn}} = 0.97 \cdot \bar{C}_{\text{ReLU}^2}$, which simultaneously shrinks `.c5` by `-254,170 B` per copy vs. Iteration 6 and improves full-benchmark forward-pass cross-entropy by `-0.4801 bpb` (`base`) / `-0.0349 bpb` (`fw`) vs. the unquantized Iteration 3 reference and `-0.2414 bpb` (`base`) / `-0.0612 bpb` (`fw`) vs. Iteration 6!), coupled linear-bottleneck KDA gate projections (`blocks.0..15` `forget_gate_projection.{up,down}` and `output_gate_projection.{up,down}` with Lehmer re-permutation), channel-weighted `attention.{query,key,value,output}_projection.weight.q` (`blocks.0..15`), and `embedding.*` / `value_embeddings.0..3` / `prior_embedding` snapping onto the true 4-level QAT grid `{0, ±2, ±4, ±7}` (`tf1_np_codec qat-snap`, preserving 100 % of `npkey` Lehmer digits `K = 20,277 B`) — shrinking the container from `3,744,629 B` to **`2,789,635 B`** (`-954,994 B` per copy / **`-1,909,988 B` in `S`** vs `cmix-neif-2.0`; `-254,170 B` per copy / `-508,340 B` in `S` vs Iter 6; `-943,490 B` per copy / `-1,886,980 B` in `S` vs Iter 3; FNV-1a-64 `0xef3bf4aeffc73099ULL`, SHA-256 `3c82e28426cb7e1d74ce389b34d6f200704553bc8f38c022328a62f23afc9479`).
3. **BITLSTM compact 4-bit exponent+nibble weight packing, active-cell `H` fast-path, `tf1` SIMD kernel deduplication & dead-code pruning (`src/bit_head_weights.h`, `src/bit_head.cpp`, `src/tf1/opt/qmat_dense.cpp`, `src/tf1/opt/qmat_sparse.cpp`, `src/mixer_precond.cpp`, `src/mixer/sse.cpp`, `src/glibc_compat.cpp`, `src/models/match.cpp`)**: rounded slot-1 (`i in [2561, 13570)`) `bf16` mantissas from 4 bits to 3 bits (round-to-nearest-even) and packed each weight into a single byte `(e_code << 4) | s` in `kBlobW[24669]` (`e_code = e - 116` for `e in [116, 130]` covering `99.63%` of all `24,579` weights, and `e_code = 15` with a 90-byte exception tail `kBlobW[24579..24668]` indexed by `kSlotExcOff[3] = {0, 4, 55}`), cutting `kBlobW` `.rodata` in half from `49,158 B` to `24,669 B` (`-24,489 B` raw `.rodata`, `-49.8%`, `-19,568 B` stripped ELF `Nf`/`Nd`); pruned all dead online-training functions and 39 unused pointer/scalar fields from `struct Head` and `Head::Free()` (`g_head` in `.bss` shrunk to `264 B`); restricted `Head::Forward()` to active cells `H = kSlotH[slot]` (`H = 16` via `Dot16Preloaded` (`Hsum4`) for slot 0 across all 30.99M metadata/census bits, cutting slot-0 `BITLSTM` compute by `~2.7x`, and `H = 32` via `Dot32Preloaded` (`Hsum8`) for slots 1 and 2); transposed `hidP[cc * nhid + r]` to vectorize `Adjust()` in non-FMA AVX2 with `0 ULP` difference against scalar IEEE-754 order; added `if (!mk) continue;` and `if (!m) continue;` zero-block fast-paths in `relu2q_phase2` (`qmat_dense.cpp`) and `qsparse_make_idx` (`qmat_sparse.cpp`), deduplicated `qgemv_relu2q` and `qgemv_quant_bias`, and folded the odd-`nnz` tail into the main `sparse4_accum` loop (`npairs = (nnz + 1) / 2`), shrinking stripped ELF `Nf`/`Nd` by another `-1,408 B` and `.decomp_bin` by `-588 B` while avoiding a second 96-instruction AVX2 flush on 17/18 of odd-`nnz` calls; vectorized `mixer_precond.cpp` column scaling in AVX2; removed dead functions/shims (`sse.cpp`, `glibc_compat.cpp`), replaced `history_.at()` with `history_[]` in `match.cpp`, and annotated cold lifecycle routines with `COLD_SIZE_ATTR`.
4. **PPMd page pool, SEMCACHE, `word_label` & decompressor/postprocessor lifecycle & allocators (`src/models/ppmd_page.h`, `src/models/ppmd_page.cpp`, `src/models/ppmd.cpp`, `src/semantic_cache.h`, `src/predictor.cpp`, `src/word_label.cpp`, `src/runner.cpp`, `src/arena_alloc.h`, `src/tf1/opt/arena_build.cpp`, `src/preprocess/preprocessor.cpp`, `src/readalike_prepr/komm_transform.h`, `src/readalike_prepr/page_form.h`, `src/readalike_prepr/phda9_preprocess.h`, `src/readalike_prepr/cmtidx_transform.h`, `src/readalike_prepr/census_transform.h`, `src/readalike_prepr/article_reorder.h`, `src/readalike_prepr/misc.h`)**: reduced `DYN_KLEMME_KIB` to `3.5 GiB`, max PPMd pool to `1 GiB` (`64 MiB` helper), and initial `StartSubAllocator` buffer to `256 MiB` (`64 MiB` helper); fixed `zonen_textrahmen(N)` rounding granularity from `16384` to `1024` frames (`4 MiB`) and removed the phantom `+2.12 GiB` `ppmd_page::raum(kSemTotal)` pre-charge so `PPMd` retains its full `1 GiB` pool under the `3.5 GiB` clamp; fixed the `69 MiB` `wordlbl_` memory leak in `Predictor::FreeFxcmRest()` and removed `PAGE_REGISTRY_ADD` on transient `word_label` / `SemTable` buffers to eliminate dangling pointers in `pageregistry::T()` and premature `MADV_COLLAPSE` on 50.3 MiB of untouched zero pages; kept exact `SEMCACHE` geometry (`kNLog = 24, kB = 18`) in a single demand-paged `MAP_ANONYMOUS | MAP_NORESERVE` mapping unmapped in `FreeFxcmRest()`, switched `arena::Bump::NeuerBlock()` to demand-paged 4 KiB `MAP_NORESERVE` blocks, removed redundant `memset` calls on kernel-zeroed mappings, and unrolled `SemDotPm1<N>` 4 rows at a time in AVX2 — lowering full-run peak RAM by `2.86x` (`9.77 GiB` -> `~3.34 GiB`). In Iterations 5–7, fixed the dangling `dict` pointer in `preprocessor::Decode` (`dict = NULL` after `delete dict`), eliminated the `~105 MiB` `uint64_t* line_off` array in `article_reorder.h::loadFile` by storing direct byte offsets `[start, end)` in `line_buf`, added `comment_form::Freigeben()` (`komm_transform.h`), `g1_free()` (`page_form.h`, `phda9_preprocess.h`), and explicit buffer deallocations at the end of `cmtidx_inplace` (`cmtidx_transform.h`) and `census_file` (`census_transform.h`) plus `fclose(dictionary)` in `runner.cpp` and removal of dead `revts_key`/`revts_val` vectors — cutting peak preprocessor/postprocessor heap by `~161 MiB` before `sort()` in `RunDecompression` — and vectorized `split_by_lines` (`misc.h`), `loadFile` (`article_reorder.h`), `cmtidx_count_lines` (`cmtidx_transform.h`), and `comment_form::ProcessFile` (`komm_transform.h`) with `memchr`/`memmem`/`fread`/`fwrite`.
5. **Compiler warning cleanups, build flags & attribution (`makefile`, `src/`, `AUTHORS`)**: resolved `-Wdangling-else`, `-Wbitwise-instead-of-logical`, `-Wparentheses`, `-Wconstant-logical-operand`, `-Wunused-private-field`, `-Wunused-function`, and `-Wunused-variable` diagnostics across `src/`, added `-Wno-backend-plugin` to `prof_use` and `cmix_aux` in `makefile`, added `SPDX-License-Identifier: GPL-3.0-only` modification headers to touched files, and documented all seven optimization iterations in `en-optimization_report_it_1.md`..`en-optimization_report_it_7.md`, `uk-optimization_report_it_1.md`..`uk-optimization_report_it_7.md`, and `AUTHORS`.

## Verified `cmix-neif-3.1` summary against `cmix-neif-2.0`

| metric / component | `cmix-neif-2.0` | `cmix-neif-3.1` (Iter 7) | delta |
|---|---:|---:|---:|
| `.c5` weight container (per file / in `S`) | 3,744,629 B / 7,489,258 B | **2,789,635 B** / **5,579,270 B** | **-954,994 B** / **-1,909,988 B** (`-254,170 B` / `-508,340 B` vs Iter 6; `-943,490 B` / `-1,886,980 B` vs Iter 3) |
| `BITLSTM` `kBlobW` `.rodata` & stripped ELF (`Nf` / `Nd`) | 49,158 B (`619,918` / `538,880` B) | **24,669 B** (`600,350` / `519,248` B) | **-24,489 B `kBlobW` (-49.8 %)** / **-19,568 B ELF** (`-20,160 B` vs Iter 3; `-1,408 B` vs Iter 6) |
| Packed compressor + decompressor (`cmix_orig + .decomp_bin`) | 395,323 B (211,438 + 183,885) | **397,231 B** (212,388 + 184,843) | `+1,908 B in S` (`.decomp_bin` `-588 B` vs Iter 6; includes multi-model `.c5` decoder & decompressor safety/cleanup fixes yielding `-1,909,988 B` in `.c5`) |
| Shipped `comp9` / projected `S = comp9 + archive9` | 4,042,564 B / 93,388,956 B | **3,088,520 B** / **91,480,030 B** | **-954,044 B** on `comp9` / **-1,908,926 B in `S`** (`-508,784 B` vs Iter 6; `-1,885,642 B` vs Iter 3) |
| Peak resident set on `enwik9` | 9.77 GiB | **~3.34 GiB** | **-6.20 GiB (2.86x lower)** (`-161 MiB` postprocessor heap & `-792 KiB` `ModelleId`) |
| Packer build peak RSS & single-pass time (`build_packer.sh`) | 9.12 GiB / >300 s (`systrap` trap) | **1.51 GiB** / **~38.0 s** (`201.0 s` full `build.sh`) | **6.05x lower RAM**, `4.06x` faster vs Iter 2 |
| Single-core coding / decoding speed (slices) | 1.00x | **+11.8 % to +12.6 %** (`~2.7x` faster `BITLSTM` slot 0 + sparse/ReLU² zero-block fast-paths + vectorized `memchr`/`memmem` I/O) | 100 % bit-identical round-trip |

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
| **`S` = `comp9` + `archive9` + `comp9.args` (`cmix-neif-2.0`)** | **93,388,956** *(before `-1,400,142 B` weight/code/packer savings in `cmix-neif-3.1` -> **`91,988,814 B`**)* |
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
