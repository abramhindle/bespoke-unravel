#ifndef AUTOMATON_LM_HH_
#define AUTOMATON_LM_HH_

#include "lm.hh"
#include "kenlm_with_vocab.hh"

class AutomatonLM : public LM {
 public:
  typedef size_t state_t;
  AutomatonLM(KenLmWithVocab& klm);
  virtual ~AutomatonLM();
  void build_automaton_lm(const KenLmWithVocab& klm);
  void check_automaton_lm(const KenLmWithVocab& klm, size_t n_queries);
  score_t getScore(state_t state, word_id_t word,
                   state_t* out_state) const;
  size_t getSize() const;
  state_t getNullContextState() const;

 private:
  struct StateAndScore {
    state_t state;
    score_t score;
  };
  // automaton_[state][word] = {out_state, score}
  std::vector<std::vector<StateAndScore>> automaton_;
};

#endif /* AUTOMATON_LM_HH_ */
