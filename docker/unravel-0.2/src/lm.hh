#ifndef LM_HH_
#define LM_HH_

#include "vocab.hh"


class LM {
 public:

  const Vocab& getVocabConst() const {
    return *vocab_;
  }

  Vocab& getVocabNonConst() {
    return *vocab_;
  }

 protected:
  Vocab* vocab_;
};


#endif // LM_HH_
