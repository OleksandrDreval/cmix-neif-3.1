# Changes in `cmix-neif-3.1` (against `cmix-neif-2.0`)

SPDX-License-Identifier: GPL-3.0-only

* **Author of `cmix-neif-3.1` modifications:** **Oleksandr Dreval** (<https://github.com/OleksandrDreval>)
* **Repository:** <https://github.com/OleksandrDreval/cmix-neif-3.1>
* **Copyright & License:** Copyright (C) 2026 Oleksandr Dreval — distributed under **GNU General Public License v3.0 only (`GPL-3.0-only`)**, an official OSI-Approved Open Source License (<https://opensource.org/license/gpl-3-0>) and FSF Free Software License (GPL-3.0 section 5a modification notice; see `LICENSE` and `AUTHORS`):

1. **Packer & self-extracting stub (`tools_src/packer/stub.c`, `tools_src/packer/build_packer.sh`, `build_and_construct_comp.sh`)**: moved the stage-2 warm-up `memcpy` and `x86t_range` transform ahead of the first `syscall` in `s2_main()` so user-space syscall interception (`systrap`) cannot corrupt RWX `s2_base` bytes before warm-up; upgraded host packer tools to `-O3 -march=native` and parallelized stub builds and all 6 packer cross-checks.
2. **Lossless `.c5` transformer weight container (`weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5`, `src/tf1/weights_io_compressed.cpp`, `src/tf1/weights_io.h`, `src/transformer.h`, `training/tf/3_train/export/np2/tf1_np_codec.cpp`)**: added Sign-Magnitude coding for `ENC_INT4W`, per-canonical-name (`ci`) probability tables, refined width bucketing and `CNT_LIMIT_INT4W = 1535`, per-`ci` counting hi-byte trees for `ENC_BF16C`, and `last_zrow` scale skipping, shrinking the container losslessly from `3,744,629 B` to `3,738,126 B` (`-6,503 B` per copy, `-13,006 B` in `S`, FNV-1a-64 `0x7b5d6d235c69c947ULL`).
3. **BITLSTM weights & frozen AVX2 fast-path (`src/bit_head_weights.h`, `src/bit_head.cpp`)**: rounded slot-1 (`i in [2561, 13570)`) `bf16` mantissas from 4 bits to 3 bits (round-to-nearest-even) to reduce packed binary entropy (`-1,567 B` per binary, `-3,134 B` in `S`); skipped dead backprop allocations/copies when `frozen == 1`, eliminated 24 per-bit `_mm256_storeu_ps` gate stores in `Head::Forward()`, and vectorized `hidP` projection in `Adjust()` with `DotPadded`.
4. **PPMd page pool & SEMCACHE memory / SIMD (`src/models/ppmd_page.h`, `src/models/ppmd_page.cpp`, `src/models/ppmd.cpp`, `src/semantic_cache.h`, `src/predictor.cpp`)**: reduced `DYN_KLEMME_KIB` to `3.5 GiB`, max PPMd pool to `1 GiB` (`64 MiB` helper), initial `StartSubAllocator` buffer to `256 MiB` (`64 MiB` helper), and `SEMCACHE` dimensions to `kNLog = 23, kB = 17` (`2.12 GiB` -> `1.08 GiB`), lowering full-run peak RAM by 2.86x (`9.77 GiB` -> `~3.34 GiB`); removed redundant `memset` on `MAP_ANONYMOUS` allocations and unrolled `SemDotPm1<N>` 4 rows at a time in AVX2.
5. **Compiler warning cleanups & attribution**: resolved `-Wdangling-else`, `-Wbitwise-instead-of-logical`, `-Wparentheses`, `-Wconstant-logical-operand`, `-Wunused-private-field`, `-Wunused-function`, and `-Wunused-variable` diagnostics across `src/`, added `SPDX-License-Identifier: GPL-3.0-only` modification headers to touched files, and documented benchmarks in `en-optimization_report.md`, `uk-optimization_report.md`, and `AUTHORS`.

## Verified `cmix-neif-3.1` summary against `cmix-neif-2.0`

| metric / component | `cmix-neif-2.0` | `cmix-neif-3.1` | delta |
|---|---:|---:|---:|
| `.c5` weight container (per file / in `S`) | 3,744,629 B / 7,489,258 B | **3,738,126 B** / **7,476,252 B** | **-6,503 B** / **-13,006 B** |
| `BITLSTM` `kBlobW` packed x86-CM (per binary / in `S`) | 22,728 B / 45,456 B | **21,161 B** / **42,322 B** | **-1,567 B** / **-3,134 B** |
| Peak resident set on `enwik9` | 9.77 GiB | **~3.34 GiB** | **-6.35 GiB (2.86x lower)** |
| Single-core coding / decoding speed (slices) | 1.00x | **+10.4 % to +10.7 %** | 100 % bit-identical round-trip |
| Packer build & 6 bit-exact cross-checks | >300 s / `systrap` failure | **143.0 s** | **>2.1x faster**, `systrap`-safe |

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
| **`S` = `comp9` + `archive9` + `comp9.args` (`cmix-neif-2.0`)** | **93,388,956** *(before `-16,140 B` weight/blob savings in `cmix-neif-3.1`)* |
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
