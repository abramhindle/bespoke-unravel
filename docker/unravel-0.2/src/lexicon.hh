#ifndef LEXICON_H_
#define LEXICON_H_

#include <vector>
#include <boost/iostreams/filtering_stream.hpp>
#include <set>

#include "global.hh"
#include "vocab.hh"

class Lexicon {
 public:
  Lexicon() {};
  Lexicon(Vocab* eVocab, Vocab* fVocab, size_t beamSize);

  virtual ~Lexicon() {};

  virtual void add(word_id_t e, word_id_t f, score_t score) = 0;
  virtual size_t addCandidates(word_id_t f, std::set<word_id_t>* candidates,
                               size_t limit = 10000) const = 0;
  virtual score_t score(word_id_t f, word_id_t e) const = 0;
  virtual void init() = 0;
  virtual void learn() = 0;
  virtual void printLex(score_t thresh) const = 0;

  virtual void write(std::string fn) const = 0;
  virtual void read(std::string fn, bool add_words = false) = 0;
};

#endif /* LEXICON_H_ */
