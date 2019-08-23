#ifndef ACTUALSTRINGSKENLM_H_
#define ACTUALSTRINGSKENLM_H_

#include "vocab.hh"

#include <lm/enumerate_vocab.hh>
#include <lm/word_index.hh>

#include <vector>

/**
 * Receives callbacks from KenLM to build vocab
 */
class ActualStringsKenLm: public lm::EnumerateVocab {
  public:
    ActualStringsKenLm(Vocab& vocab);
    virtual ~ActualStringsKenLm();
    virtual void Add(lm::WordIndex index, const StringPiece &str);
    Vocab getVocab() const;
  private:
    Vocab& vocab;
};

#endif /* ACTUALSTRINGSKENLM_H_ */
