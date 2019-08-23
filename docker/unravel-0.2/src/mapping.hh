#ifndef MAPPING_H_
#define MAPPING_H_

#include "global.hh"
#include "vocab.hh"
#ifdef WITH_OPENFST
#include <fst/fstlib.h>
#include <fst/vector-fst.h>
#endif

#include <boost/iostreams/filtering_stream.hpp>
#include <vector>
#include <set>

#include "extension_order.hh"
#include "reference_mapping.hh"


class ExtensionOrder;
class ReferenceMapping;
struct partial_hyp_body;

class Mapping {
 public:
  enum AddingCode {
    CONSISTENT_NOT_ADDED = 0,
    SUCC_ADDED = 1,
    INCONSISTENT_NOT_ADDED = 2
  };

  // helper method
  static void hypToMapping(const partial_hyp_body & source_hyp,
                           Mapping* target_mapping, size_t card);

  Mapping(size_t vocab_size_f, size_t vocab_size_e);
  virtual ~Mapping();

  void setScore(score_t score);
  double getScore(void) const;
  int getCardinality(void) const;
  std::vector<word_id_t> applyMapping(
      std::vector<word_id_t> cipherVector) const;
  size_t correctCount(const Mapping& reference) const;

  void writeLexicon(Vocab const &eVocab, Vocab const &fVocab,
                    boost::iostreams::filtering_ostream* os) const;
  void readLexicon(Vocab const &eVocab, Vocab const &fVocab,
                   boost::iostreams::filtering_istream* is);
#ifdef WITH_OPENFST
  void fstFgivenE(fst::VectorFst<fst::StdArc> * lexFst, const Vocab &eVocab,
                  const Vocab &fVocab, score_t lambda);
#endif

  void dumpInfo(Vocab const &eVocab, Vocab const &fVocab, std::ostream * os,
                bool compact = false) const;
  void dumpInfo(Vocab const &eVocab, Vocab const &fVocab,
                boost::iostreams::filtering_ostream* os,
                bool compact = false) const;
  void printInfo() const;
  void printInfo(const Vocab& eVocab, const Vocab& fVocab,
                 const Mapping & reference) const;
  std::string dumpToString(const Vocab& eVocab, const Vocab& fVocab,
                    const ExtensionOrder * extOrder) const;

  void printRatioCorrectWrong(const ReferenceMapping* reference,
                              std::ostream* os) const;

  inline bool hasE(size_t e) const {
    // make 0 the special value, then just return the value (no additional if)
    return countE[e] > 0;
  }

  inline bool hasF(size_t f) const {
    CHECK_LT(f, mappingFToE.size());
    return mappingFToE[f] != MAPPING_NOT_MAPPED;
  }

  // just overwrite
  inline void set_(word_id_t e, word_id_t f) {
    mappingFToE[f] = e;
  }

  // swap mappings of cipher words
  inline void swap_(word_id_t f, word_id_t ff) {
    word_id_t old = mappingFToE[f];
    mappingFToE[f] = mappingFToE[ff];
    mappingFToE[ff] = old;
  }

  inline bool mapNgram_(const word_id_t * cipher, word_id_t * plain,
                        const order_t order) const {
    bool full = true;
    for (order_t i = 0; i < order; ++i) {
      if (!this->hasF(cipher[i])) {
        full = false;
        break;
      } else {
        plain[i] = this->getE(cipher[i]);
      }
    }
    return full;
  }

  void reset() {
    for (size_t f=0;f<mappingFToE.size();++f) mappingFToE[f] = MAPPING_NOT_MAPPED;
    for (size_t e=0;e<countE.size();++e) countE[e] = 0;
    cardinality = 0;
    score = 0.0;
  }

  // raw add, no checks
  AddingCode add_(word_id_t e, word_id_t f) {
    mappingFToE[f] = e;
    // keep track of how often plaintext words have been mapped
    if (countE[e]+1 >= countMapE.size()) countMapE.resize(countE[e]+2);
    if (countE[e] > 0) countMapE[countE[e]].erase(e);
    countMapE[++countE[e]].insert(e);

    // total cardinality increases
    ++cardinality;
    return SUCC_ADDED;
  }

  // add with checks
  inline AddingCode add(word_id_t e, word_id_t f) {
    // did e already map to something?
    bool f_in = countE[e] != 0;
    // did f already map to something?
    word_id_t e_in = mappingFToE[f];
    if (!f_in && e_in == MAPPING_NOT_MAPPED) {
      // both not mapped
      return add_(e, f);
    } else if (f_in && e_in == e) {
      // consistent, but not added
      return CONSISTENT_NOT_ADDED;
    } else {
      // not added, adding would cause inconsistency
      return INCONSISTENT_NOT_ADDED;
    }
  }

  inline const std::vector<std::set<word_id_t> > & getCountMap() const {
    return countMapE;
  }

  inline size_t getESize() const {
    return countE.size();
  }

  inline size_t getFSize() const {
    return mappingFToE.size();
  }

  inline size_t getNumberOfMappings(size_t p) const {
    CHECK_LT(p, countE.size());
    return countE[p];
  }

  inline void removeMapping(word_id_t p, word_id_t c) {
    CHECK_GT(countE[p], 0);
    countMapE[countE[p]].erase(p);
    --countE[p];
    if (countE[p] > 0) countMapE[countE[p]].insert(p);
    // add CHECKs
    mappingFToE[c] = MAPPING_NOT_MAPPED;
    --cardinality;
  }

  inline size_t getE(size_t f) const {
    CHECK_LT(f, mappingFToE.size());
    return mappingFToE[f];
  }

  struct MappingCompare {
    bool operator()(Mapping* lhs, Mapping* rhs) {
      return lhs->getScore() < rhs->getScore();
    }
  };

  static bool biggerRelationForPointers(Mapping* lhs, Mapping* rhs) {
    return lhs->getScore() > rhs->getScore();
  }

 private:
  // countMapE[i] contains all words mapped i times
  std::vector<std::set<word_id_t> > countMapE;
  std::vector<word_id_t> mappingFToE;
  std::vector<count_t> countE;  // TODO(anybody): should not be in mapping class
  int cardinality;
  score_t score;
};

bool operator<(Mapping const &lhs, Mapping const &rhs);

#endif /* MAPPING_H_ */
