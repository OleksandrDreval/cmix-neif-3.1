#include "src/tf1/opt/model_opt.h"
#include "src/tf1/tf1_prior0.h"
#include "src/vocab_frozen.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace fx2::opt;

struct CeResult {
  double base_bpb;
  double fw_bpb;
  size_t tokens;
};

CeResult eval_stream(TransformerOpt& opt, const char* path, int max_tokens = -1) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> data(sz);
  if (fread(data.data(), 1, sz, f) != (size_t)sz) { perror("fread"); exit(1); }
  fclose(f);

  std::vector<uint8_t> tokens;
  for (unsigned char b : data) {
    if (vocab_frozen::InSet(b)) {
      tokens.push_back(vocab_frozen::kByteMap[b]);
    }
  }
  if (max_tokens > 0 && tokens.size() > (size_t)max_tokens) {
    tokens.resize(max_tokens);
  }

  opt.begin_article();
  const uint16_t prior0[208] = PRIOR0_INIT;
  alignas(64) float probs[208];
  double base_nll = 0.0;
  double fw_nll = 0.0;
  size_t count = 0;

  for (size_t i = 0; i < tokens.size(); ++i) {
    uint8_t tok = tokens[i];
    opt.step(tok, prior0, probs);
    const float* fw_probs = opt.fw_probs();
    if (i + 1 < tokens.size()) {
      uint8_t target = tokens[i + 1];
      float p_base = std::max(probs[target], 1e-12f);
      float p_fw = std::max(fw_probs[target], 1e-12f);
      base_nll += -std::log2(p_base);
      fw_nll += -std::log2(p_fw);
      count++;
      opt.fw_update(target);
    }
  }

  double base_bpb = count > 0 ? (base_nll / count) : 0.0;
  double fw_bpb = count > 0 ? (fw_nll / count) : 0.0;
  return {base_bpb, fw_bpb, count};
}

int main(int argc, char** argv) {
  const char* weights = "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5";
  int max_tokens = -1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--short")) {
      max_tokens = 4096;
    } else if (!strncmp(argv[i], "--tokens=", 9)) {
      max_tokens = atoi(argv[i] + 9);
    } else if (argv[i][0] != '-') {
      weights = argv[i];
    }
  }

  printf("Loading transformer weights: %s ...\n", weights);
  TransformerOpt opt(weights, AttnKind::KVI8);

  printf("--- Evaluating prof_input/input (max_tokens=%d) ---\n", max_tokens);
  CeResult r1 = eval_stream(opt, "prof_input/input", max_tokens);
  printf("prof_input/input (N = %zu tokens): base_bpb = %.6f, fw_bpb = %.6f\n",
         r1.tokens, r1.base_bpb, r1.fw_bpb);

  int in2_tokens = max_tokens;
  if (max_tokens == -1) {
    in2_tokens = 28668; // 44,700 + 28,668 = 73,368 tokens benchmark
  }
  printf("--- Evaluating prof_input/input2 (max_tokens=%d) ---\n", in2_tokens);
  CeResult r2 = eval_stream(opt, "prof_input/input2", in2_tokens);
  printf("prof_input/input2 (N = %zu tokens): base_bpb = %.6f, fw_bpb = %.6f\n",
         r2.tokens, r2.base_bpb, r2.fw_bpb);

  double mean_base = (r1.base_bpb + r2.base_bpb) / 2.0;
  double mean_fw = (r1.fw_bpb + r2.fw_bpb) / 2.0;
  printf("--- Benchmark Summary ---\n");
  printf("Mean: base_bpb = %.6f, fw_bpb = %.6f\n", mean_base, mean_fw);

  if (mean_fw > 5.0) {
    fprintf(stderr, "Cross-entropy regression detected (mean_fw > 5.0)!\n");
    return 1;
  }
  printf("Cross-entropy verification PASSED.\n");
  return 0;
}
