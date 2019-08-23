#ifndef KENLMWITHVOCAB_H_
#define KENLMWITHVOCAB_H_

#include "lm.hh"
#include "ngram.hh"

#include <sstream>
#include <cmath>
#include <set>
#include <lm/word_index.hh>
#include <lm/model.hh>


////////////////////////////////////////////////////////////////////////////
// Wrapper around KenLM for having access to vocabulary and probabilities
// directly based on word index
////////////////////////////////////////////////////////////////////////////
class KenLmWithVocab : public LM {
 public:
  typedef lm::ngram::State state_t;
  KenLmWithVocab();
  KenLmWithVocab(const std::string &file, Vocab* vocab);
  void read(const std::string &file, Vocab* vocab);
  virtual ~KenLmWithVocab();
  lm::ngram::Model const & getModel() const;
  state_t getState(const state_t & state, word_id_t word) const;
  score_t getUnigramScore(word_id_t word) const;
  state_t getNgramState(const std::vector<word_id_t> & ngram) const;
  state_t getNullContextState() const;
  std::string stateToString(const state_t &state) const;
  std::string stateToStringNormalOrder(const state_t &state) const;
  void addCandidates(
      const state_t &state, std::set<word_id_t> *candidates,
      size_t max_candidates) const;
  size_t getFirstInvalidVocabId();
  bool validWordId(word_id_t word_id) const;
  void prepare_known_states();
  void prepare_successors(size_t beam_size);


  //////////////////////////////////////////////////////////////////////////
  // inline functions in header which are called extremely often
  //////////////////////////////////////////////////////////////////////////

  inline score_t getScore(
      const state_t& state, word_id_t word, state_t* out_state) const {
    return model->Score(state, word, *out_state);
  }

  inline score_t getScore(const state_t& state, word_id_t word) const {
    state_t dummy_state;
    return model->Score(state, word, dummy_state);
  }

  inline double ngramScore(const word_id_t * ngram, order_t order) const {
    state_t state = model->NullContextState();
    state_t out_state;
    if (order > 1) {
      for (order_t i=0; i<order-1; ++i) {
        model->Score(state, ngram[i], out_state);
        state = out_state;
      }
    }
    return model->Score(state, ngram[order-1], out_state);
  }

 private:
  lm::ngram::Model *model;
  std::set<state_t> known_states_;
  std::map<state_t, std::vector<word_id_t> > best_successors_;
  word_id_t first_invalid_vocab_id_ = 0;
};

#endif // KENLMWITHVOCAB_H_
