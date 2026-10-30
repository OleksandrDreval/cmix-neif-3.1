#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "mixer/sigmoid.h"
#include "mixer/mixer-input.h"
#include "mixer/mixer.h"
#include "mixer/byte-mixer.h"
#include "mixer/sse.h"
#include "models/model.h"
#include "models/byte-model.h"
#include "context-manager.h"
#include "models/direct.h"
#include "models/direct-hash.h"
#include "models/indirect.h"
#include "models/match.h"
#include "models/grammar-match.h"
#include "models/ppmd.h"
#include "models/bracket.h"
#include "models/fxcmv1.h"
#include "contexts/context-hash.h"
#include "contexts/bracket-context.h"
#include "contexts/sparse.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval.h"
#include "contexts/interval-hash.h"
#include "contexts/bit-context.h"
#include "contexts/combined-context.h"

#include "id_exclude.h"
#include "page_contexts.h"
#include "transformer.h"
#include "tf1/tf1_engine.h"
#include "char_rate.h"
#include "run_mode.h"
#include "bit_head.h"
#include "adaptive_depth.h"
#include "semantic_cache.h"
#include "word_label.h"
#include "mixer_precond.h"

#include "ds/SmallVector.h"
#include "ds/emhash_set.hpp"

#include <vector>
#include <set>
#include <memory>
#include <optional>
#include <array>

class Predictor {
 public:
  Predictor(const std::vector<bool>& vocab);
  float Predict();
  void Perceive(int bit);
  void Pretrain(int bit);
  int ChrForced() const { return char_rate::Forced(manager_.bit_context_); }
  void FreeFxcmMemory();
  void FreeFxcmRest();
  void ObserveKnownBit(int bit);
  void ObserveTfOnlyByte(unsigned int byte);

  const float* bitl_hid_ = nullptr;
  int bitl_n_hid_ = 0;
  float bitl_tf_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int bitl_n_tf_ = 0;
  const unsigned int* bitl_zone_ = nullptr;
  const float* bitl_stage1_ = nullptr;   // layers_[1].Inputs(); this hand-over follows bitlstm32 of cmix-obias (David Freelan, GPL-3)
  int bitl_n_stage1_ = 0;
  float bitl_m1raw_ = 0.0f;
  float bb_z_ = 0.0f, bb_ppre_ = 0.5f;
  unsigned int BbBitCtx() const { return manager_.bit_context_; }
  int bitl_override_ = 0;
  bool AdepthSkipped() const { return adepth_.skip; }
  int Tf1ZwangByte() const { return tf1_engine_ ? tf1_engine_->ZwangByte() : -1; }
  void WordlblBlockEnde() { wordlbl_ = word_label::Neu(wordlbl_); }
  void MetalrBlockEnde();
  void PpmdWaermen();
  void SemByteStart(int pb);
  uint32_t SemCorrect(uint32_t p32);
  void SemLearn(int bit);
  // the semantic-cache columns of the BITLSTM row (Predict(), every bit)
  void SemAppend();

 private:
  void SemInit();
  void SemUtByte(int topb);
  uint32_t SemUtCorrect(uint32_t p32);
  void SemUtLearn(int bit);
  // the cache's readout of the bit about to be predicted (Predict(), before
  // the layer-0 row), its layer-0 input and its end-mixer column
  void SemPredBit();
  void SemSetInput(unsigned int i);
  void SemSetStage1();
  unsigned int sem_index_ = 0;
  void TreffByteStart(int last);
  uint32_t TreffCorrect(uint32_t p32);
  void TreffLearn(int bit);
  semantic_cache::State sem_;
  cache_hit::State treff_;
  adaptive_depth::State adepth_;
  unsigned long long GetNumModels();
  void AddMixer(int layer, const unsigned long long& context,
      float learning_rate);
  void AddAuxiliary();
  void AddPPMD();
  void AddBracket();
  void AddWord();
  void AddDirect();
  void AddMatch();
  void AddGrammarMatch();
  void AddDoubleIndirect();
  void AddMixers();
  void E2EGrad(int bit);

  std::unique_ptr<transformer::Tf1Engine> tf1_engine_;
  std::unique_ptr<transformer::HalfFileWriter> tf1_prior_writer_;
  std::vector<float> tf1_prior_row_;
  std::vector<int> tf1_vocab_bytes_;
  llvm::SmallVector<Indirect<Nonstationary>, 32> indirect_ns_models_; // non-stationary
  llvm::SmallVector<Indirect<RunMap>, 1> indirect_r_models_; // run map
  llvm::SmallVector<Direct, 4> direct_models_;
  llvm::SmallVector<Match, 10> match_models_;
  unsigned int tf1_match_base_ = 0;
  unsigned int tf1_match_count_ = 0;
  std::optional<GrammarMatch> grammar_match_;

  std::optional<Bracket> bracket_model_;
  // fxcm is the one "auxiliary" input (the byte LSTM was the second).
  size_t auxiliary_size_ = 1; // 0 -> fxcm
  SSE sse_;
  llvm::SmallVector<MixerInput,2> layers_;
  // Inline capacities cover every line (emplace_back stops the program
  // instead of growing): 23 layer-0 mixers, 2 final mixers (the second one
  // holds the add rows).
  llvm::SmallVector<Mixer, 23> mixer_0_;
  llvm::SmallVector<Mixer, 2> mixer_1_;
  std::vector<unsigned int> auxiliary_;
  ContextManager manager_;
  Sigmoid sigmoid_;
  std::array<float, 4096> fxcm_stretched_inputs_;
  float fxcm_neutral_input_ = 0.0f;
  std::array<float, 4096> aux_pr_stretched_;
  std::optional<PPMD::PPMD> byte_model_;
  std::optional<Tf1Slot> tf1_slot_;
  // selector of the lx1_agree mixer from the transformer row (predictor.cpp)
  void Lx1TfConf();
  // The slot of the transformer's second (fast-weight) head and
  // its steps (predictor.cpp, noinline: no new block in Predict/Perceive).
  std::optional<Tf1Slot> fw_slot_;
  unsigned int fw_index_ = 0;   // its layer-0 input
  void FwSetInput(unsigned int i);
  void FwSetStage1();
  void FwPerceive(int bit);
  void FwSetNeutral();
  void FwSetRow();
  void FwSetOff();
  void FwPredictBit();
  // the add rows of the final mixer; every step is a noinline call in
  // predictor.cpp
  void KtxSelect();
  void KtxPrime();
  float KtxAddMix();
  void KtxW1Sum(float* w1s, unsigned int n);
  void MlPath(bool byte_update);
  // the Bayes term of the final mixer
  float MxsBayMix(float zrow);
  unsigned int mxs_ppmd_ix_ = 0;   // layer-0 index of the PPMd input
  // the final-mixer output z of this bit (all rows plus the Bayes term); every
  // final-mixer row learns on the error of z
  float end_z_ = 0.0f;
  void EndLearn(int bit);
  float sem_s1buf_[128] = {0.0f};   // the row plus the semantic-cache columns
  // the 13 columns of the BITLSTM row behind the stage-1 row (FW logit, true
  // bit position, transformer path state of MlPath); predictor.cpp, noinline.
  void MlAppend();
  float bitl_s1buf_[128] = {0.0f};   // the row plus these columns
  mixer_precond::L0Col mxp_l0col_;   // layer-0 column powers

  word_label::State* wordlbl_ = nullptr;
  // A reference: the caller (RunCompression/RunDecompression) keeps vocab
  // alive longer than the Predictor, so no copy of it is made.
  const std::vector<bool>& vocab_;
   FXCM fxcm_model_;

  friend unsigned long long PredictorLayoutTagFast();
  unsigned int last_p16_ = 32768;
};

unsigned long long PredictorLayoutTagFast();

#endif
