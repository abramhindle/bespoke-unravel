#ifndef COOC_HH_
#define COOC_HH_

#include <boost/iostreams/filtering_stream.hpp>
#include <set>

#include "vocab.hh"
#include "extension_order.hh"
#include "ngram.hh"

#define CLASS_NOT_FOUND 9999999

class ExtensionOrder;

class Cooc {
 public:
  Cooc();
  void read(boost::iostreams::filtering_istream& cooc_is, Vocab& vocab,
            bool verbose);
  const std::set<word_id_t>* getCooc(word_id_t word_id) const;
  void addCooc(word_id_t e, word_id_t e_prime);
  virtual ~Cooc();
  Vocab& getVocab();
  const Vocab& getVocabConst() const;

 private:
  Vocab* vocab_;
  std::map<word_id_t, std::set<word_id_t> > cooc_map_;
  size_t max_class_;
};

#endif /* COOC_HH_ */
