#ifndef COUNTS_HH_
#define COUNTS_HH_

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <set>

#include "global.hh"
#include "extension_order.hh"
#include "ngram.hh"
#include "vocab.hh"

class ExtensionOrder;

class Counts {
 public:
  Counts(bool pad_sent_start);
  void read(std::string filename, order_t order, count_t mincount,
            Vocab& vocab, bool calc_word_to_ngrams = false);
  virtual ~Counts();
  Vocab& getVocab();
  const Vocab& getVocabConst() const;
  const std::vector<Ngram>& getNgrams() const;
  order_t getOrder() const;
  const std::vector<std::set<Ngram*> >& getWordToNgrams() const;
  const std::vector<std::set<Ngram*> >& getNewNgrams() const;
  void printNumberOfNgrams() const;
  bool initialized() { return vocab_ != 0; }
  void fillNewNgrams(const ExtensionOrder& ext_order);
  count_t getZerogramCount() const { return zerogram_count_; }
  void printCounts() const;

 private:
  void addSentenceStartCounts(const std::vector<word_id_t>& vec, double count,
                              std::map<Ngram, count_t>& ngrams_map);
  order_t order_;
  Vocab* vocab_;
  std::vector<Ngram> ngrams_;
  std::vector<std::set<Ngram*> > word_to_ngrams_;
  count_t zerogram_count_;
  std::vector<std::set<Ngram*> > newNgrams_;
  bool pad_sent_start_;
};

#endif /* COUNTS_HH_ */
