#ifndef CLASSES_HH_
#define CLASSES_HH_

#include <boost/iostreams/filtering_stream.hpp>
#include <set>

#include "vocab.hh"
#include "extension_order.hh"
#include "ngram.hh"

#define CLASS_NOT_FOUND 9999999

class ExtensionOrder;

class Classes {
 public:
  Classes();
  void read(boost::iostreams::filtering_istream& classes_is, Vocab& vocab,
            bool verbose);
  virtual ~Classes();
  Vocab& getVocab();
  const Vocab& getVocabConst() const;
  class_id_t getClass(word_id_t word_id) const;
  size_t getMaxClass() const { return max_class_; }

 private:
  Vocab* vocab_;
  std::vector<class_id_t> class_map_;
  size_t max_class_;
};

#endif /* CLASSES_HH_ */
