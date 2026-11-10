// Based on STARLIT (Artemiy Margaritov, github.com/amargaritov/starlit,
// GPL-3.0), src/readalike_prepr/; inherited through fx2-cmix and cmix-lex.

// The comment-index transform inserts one line per affected page into .main
// (an index mark behind "</text>"), so the fixed cut lines of split4Decomp
// move by exactly that count; it follows from the total line count of the
// stream (intro and coda are untouched).  See cmtidx_transform.h.
#include "cmtidx_transform.h"

#define COMP_INTRO_END_LINE 29
#define COMP_MAIN_END_LINE  13146932
#define COMP_CODA_END_LINE  13147025

#define DECOMP_MAIN_END_LINE  13146905
#define DECOMP_INTRO_END_LINE 13146934
#define DECOMP_CODA_END_LINE  13147027

static void split_by_lines(const char* in_path, const char* p1, const char* p2,
                           const char* p3, long b1, long b2) {
  FILE* ifile = fopen(in_path, "rb");
  FILE* ofile1 = fopen(p1, "wb");
  FILE* ofile2 = fopen(p2, "wb");
  FILE* ofile3 = fopen(p3, "wb");
  long line_count = 0;
  char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), ifile)) > 0) {
    size_t pos = 0;
    while (pos < n) {
      FILE* out = (line_count < b1) ? ofile1 : (line_count < b2 ? ofile2 : ofile3);
      long lim = (line_count < b1) ? b1 : (line_count < b2 ? b2 : -1);
      size_t cur = pos;
      if (lim < 0) {
        cur = n;
      } else {
        while (cur < n && line_count < lim) {
          const char* nl = (const char*)memchr(buf + cur, '\n', n - cur);
          if (!nl) { cur = n; break; }
          cur = (size_t)(nl - buf) + 1;
          ++line_count;
        }
      }
      fwrite(buf + pos, 1, cur - pos, out);
      pos = cur;
    }
  }
  fclose(ifile);
  fclose(ofile1);
  fclose(ofile2);
  fclose(ofile3);
}

void split4Comp(char const *enwik9_filename) {
  split_by_lines(enwik9_filename, ".intro", ".main", ".coda",
                 COMP_INTRO_END_LINE, COMP_MAIN_END_LINE);
}

void split4Decomp() {
  const long sh = cmtidx_line_shift(".input_decomp");
  split_by_lines(".input_decomp", ".main_decomp", ".intro_decomp", ".coda_decomp",
                 13146906 + sh, 13146935 + sh);
}
