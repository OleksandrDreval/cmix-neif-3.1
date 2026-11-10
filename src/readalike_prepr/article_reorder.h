// Based on STARLIT (Artemiy Margaritov, github.com/amargaritov/starlit,
// GPL-3.0), src/readalike_prepr/; inherited through fx2-cmix and cmix-lex.
#ifndef ARTICLE_REORDER_H
#define ARTICLE_REORDER_H
#include "../meld.h"

#include <string.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <unordered_map>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>

#include <stdlib.h>
#include "order_codec.h"

#define NUM_OF_ARTICLES 243425

struct Accumulator {
  int id;
  size_t start;
  size_t end;
};

enum ParserState {
  expect_page = 0,
  expect_id,
  expect_pageend
};

const char *patterns1[] = { "<page>", "<id>", "</page>" };
const int transitions1[3] = { expect_id, expect_pageend, expect_page };
// pages of the loaded file: id and byte range in line_buf
static Accumulator* vec = nullptr;
static size_t vec_n = 0, vec_cap = 0;
static char* line_buf = nullptr;      // text of all lines, back to back
static size_t line_buf_len = 0;
// in phda9
int wfgets(char *str, int count, FILE  *fp);
void wfputs(const char *str,FILE *fp);

// Write byte range [a, b) of the store.
static void writeLines(size_t a, size_t b, FILE* out) {
  fwrite(line_buf + a, 1, b - a, out);
}

static void freeLines() {
  free(line_buf); line_buf = nullptr;
  line_buf_len = 0;
  free(vec); vec = nullptr;
  vec_n = vec_cap = 0;
}

void loadFile(const char *fname) {
  int state = expect_page;
  Accumulator acc = {0, 0, 0};
  FILE* sf = fopen(fname, "rb");   // size without stat (one import less)
  size_t n = 0;
  if (sf) {
    fseek(sf, 0, SEEK_END);
    n = (size_t)ftell(sf);
    fseek(sf, 0, SEEK_SET);
  }
  freeLines();
  line_buf = (char*)malloc(n + 1);  // the text never outgrows the file
  if (!line_buf) exit(1);
  if (sf) {
    if (n) n = fread(line_buf, 1, n, sf);
    fclose(sf);
  }
  line_buf[n] = 0;
  line_buf_len = n;
  static const uint8_t kPatLen[3] = {6, 4, 7};
  for (size_t pos = 0; pos < n; ) {
    size_t max_len = n - pos;
    if (max_len > 8192 * 8 - 1) max_len = 8192 * 8 - 1;
    const char* nl = (const char*)memchr(line_buf + pos, '\n', max_len);
    size_t slen = nl ? (size_t)(nl - (line_buf + pos)) + 1 : max_len;
    const char* p = (const char*)memmem(line_buf + pos, slen, patterns1[state], kPatLen[state]);
    if (p) {
      if (state == expect_page) {
        acc.start = pos;
      } else if (state == expect_id) {
        acc.id = atoi(p + 4);
      } else if (state == expect_pageend) {
        acc.end = pos + slen;
      }
      state = transitions1[state];
      if (state == expect_page) {
        if (vec_n == vec_cap) {
          vec_cap = vec_cap * 2 + 262144;
          vec = (Accumulator*)realloc(vec, vec_cap * sizeof(Accumulator));
          if (!vec) exit(1);
        }
        vec[vec_n++] = acc;
      }
    }
    pos += slen;
  }
}
// The store holds .main up to its first NUL, and .main has none: the lines
// read below are those of the file.
void reorder() {
    loadFile(".main");
    const char* const txt = line_buf;
    const char* const txt_end = line_buf + line_buf_len;

    // Article numbers need to be remapped (see article_remap.cpp): at every
    // page line the number of non-redirect pages before the previous page is
    // mapped to the index of that page (the last write wins, an unwritten
    // number maps to 0).  remap[u + 1] is the page of unit u.
    int* remap = (int*)calloc(NUM_OF_ARTICLES + 2, sizeof(int));
    {
      int count1 = -1, count2 = -1;
      bool redirect = false;
      for (const char* p = txt; p < txt_end; ) {
        const char* e = (const char*)memchr(p, '\n', (size_t)(txt_end - p));
        const size_t n = e ? (size_t)(e - p) : (size_t)(txt_end - p);
        if (ord::IsRedirLine(p, n)) redirect = true;
        if (n == 8 && memcmp(p, "  <page>", 8) == 0) {
          if (count2 + 1 <= NUM_OF_ARTICLES + 1) remap[count2 + 1] = count1;
          if (!redirect) ++count2;
          ++count1;
          redirect = false;
        }
        p += n + 1;
      }
    }

    int* units = (int*)malloc(NUM_OF_ARTICLES * sizeof(int));
    const int nu = ord::DecodeOrderV3(".new_article_order", txt, (size_t)(txt_end - txt),
                                     units, NUM_OF_ARTICLES);
    if (nu < 0) {
      msg_stop(115);
      exit(1);
    }
    int* positions = (int*)malloc((size_t)(nu + NUM_OF_ARTICLES) * sizeof(int));
    unsigned char* used = (unsigned char*)calloc(NUM_OF_ARTICLES, 1);
    int np = 0;
    for (int i = 0; i < nu; i++) {
      int res = remap[units[i] + 1];    // Article numbers get remapped.
      positions[np++] = res;
      used[res] = 1;
    }
    free(units); free(remap);

    const int n_front = np;
    if (np < NUM_OF_ARTICLES) {
        for (int i = 0; i < NUM_OF_ARTICLES; i++) {
            if (used[i] == 0) {
                positions[np++] = i;
            }
       }
    }
    free(used);

  FILE* out = fopen(".main_reordered", "wb");
  const int n_rest = np - n_front;
  for(int i = 0; i < np; i++) {
    int pos = positions[i < n_rest ? n_front + i : i - n_rest];
    writeLines(vec[pos].start, vec[pos].end, out);
  }
  fclose(out);
  free(positions);
  freeLines();
}

// pages by id ascending (the ids of enwik9 are distinct)
static int AccLess(const void* a, const void* b) {
  const int x = ((const Accumulator*)a)->id, y = ((const Accumulator*)b)->id;
  return (x > y) - (x < y);
}

void sort() {
  loadFile(".main_decomp_restored");

  qsort(vec, vec_n, sizeof(Accumulator), AccLess);

  FILE* out = fopen(".main_decomp_restored_sorted", "wb");
  for(size_t i =0; i < vec_n; i++) {
    writeLines(vec[i].start, vec[i].end, out);
  }

  fclose(out);
  freeLines();
}
#endif // ARTICLE_REORDER_H
