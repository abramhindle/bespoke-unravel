#ifndef LEXICON_SPARSE_H_
#define LEXICON_SPARSE_H_

#include "global.hh"
#include "misc.hh"
#include "vocab.hh"
#include "kenlm_with_vocab.hh"
#include <vector>
#include <boost/iostreams/filtering_stream.hpp>
#include <set>
#include "lexicon.hh"

class LexiconSparse : public Lexicon {
 public:
  LexiconSparse(Vocab* eVocab, Vocab* fVocab, size_t beamSize,
                score_t delta_lambda)
      : e_vocab_(eVocab),
        f_vocab_(fVocab),
        beamSize_(beamSize),
        delta_lambda_(delta_lambda) {
    init();
  };

  virtual ~LexiconSparse() {};

  // copy to lexicon
  void init();
  void add(word_id_t e, word_id_t f, score_t lin_score);
  size_t addCandidates(word_id_t f, std::set<word_id_t>* candidates,
                       size_t limit = 10000) const;
  score_t score(word_id_t f, word_id_t e) const;
  score_t lin_score_fast(word_id_t f, word_id_t e) const;
  void setupBeam(size_t max_vocab_id);
  const std::vector<word_id_t>& getBeam(word_id_t f) const;
  score_t ibm1(const std::vector<word_id_t>& e_sentence,
               const std::vector<word_id_t>& f_sentence);
  std::map<size_t, size_t> ibm1Map(const std::vector<word_id_t>& e_sentence,
                                   const std::vector<word_id_t>& f_sentence,
                                   score_t threshold);
  void learn();
  void learn_norm_reverse(const std::vector<score_t>& p_e,
                          const std::vector<score_t>& p_f);
  void constrain_on_e(const std::vector<word_id_t>& constrained_e, size_t n,
                      score_t boost_factor);
  void constrain_on_f(const std::vector<word_id_t>& constrained_f, size_t m,
                      score_t boost_factor);
  void write(std::string fn) const;
  void read(std::string fn, bool add_words = false);

  void printLex(score_t thresh = 0.0) const;
  void setDeltaLambda(score_t delta_lambda);

  const std::vector<std::unordered_map<word_id_t, score_t> >& getCounts() const;

  void addCountsFromLex(const LexiconSparse& lex, size_t max_evocab_id);
  void addCountsFromLex(const LexiconSparse& lex) {
    CHECK_NOTNULL(e_vocab_);
    addCountsFromLex(lex, e_vocab_->size());
  }

  void analyzeCorrectProb(const LexiconSparse& ref_lex,
                          const std::string& eval_fn) const;

  void init_id();
  void init_supervised(const std::vector<std::vector<word_id_t> >& snts,
                       const std::vector<std::vector<word_id_t> >& snts_ref);
  void init_binomial(const KenLmWithVocab& lm,
                     const std::vector<std::vector<word_id_t> >& snts);

  Vocab* e_vocab_;
  Vocab* f_vocab_;

 private:
  size_t beamSize_;
  score_t delta_lambda_;

  std::vector<std::vector<word_id_t> > lex_beam_;

  // these are sparse p(f|e) = ef_counts_[e][f]
  std::vector<std::unordered_map<word_id_t, score_t> > ef_counts_;
};

#endif /* LEXICON_SPARSE_H_ */
