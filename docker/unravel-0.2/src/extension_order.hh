#ifndef EXTENSION_ORDER_HH_
#define EXTENSION_ORDER_HH_

#include "mapping.hh"
#include "ngram.hh"
#include "misc.hh"
#include "counts.hh"
#include "embedding.hh"

#include <iostream>
#include <ostream>
#include <fstream>
#include <stddef.h>
#include <vector>
#include <boost/algorithm/string.hpp>

class Counts;
class Mapping;
class Embedding;

struct ExtOrderHyp {
public:
    ExtOrderHyp(std::vector<word_id_t>& ext_order, std::vector<bool>& words_calculated, size_t n_countss) : ext_order(ext_order), words_calculated(words_calculated), scores(n_countss, std::vector<size_t>(1, 0)) {
    }
    virtual ~ExtOrderHyp() {
    }
    std::vector<word_id_t> ext_order;
    std::vector<bool> words_calculated;
    std::vector<std::vector<size_t> > scores; // scores[counts_id][card]
};

struct doCompare { 
  doCompare( const std::vector<double>& ppl_vec ) : ppl_vec_(ppl_vec) { }
  const std::vector<double> ppl_vec_;

  bool operator()( const ExtOrderHyp* lhs, const ExtOrderHyp* rhs) {
    /*
       size_t p = lhs->scores[0].size();
       if (p>0) p--;

       double lhs_score = 0.0;
       double rhs_score = 0.0;
       for (size_t n=0; n<lhs->scores.size(); ++n) {
       lhs_score += pow(n + 1,0.3) * pow(lhs->scores[n][p],3.0);
       rhs_score += pow(n + 1,0.3) * pow(rhs->scores[n][p],3.0);
       }

       return lhs_score > rhs_score;*/

    // map total counts to effectively used counts
    // idea: a 5-gram contains multiple 2-,3-, and 4-grams
    // but they will not be used for calculating the score

    CHECK_EQ(lhs->scores.size(), rhs->scores.size());

    std::vector<size_t> lhs_counts, rhs_counts;
    size_t p = lhs->scores[0].size(); if (p>0) p--;
    for (size_t n=0;n<lhs->scores.size(); ++n) {
      CHECK_EQ(lhs->scores[n].size(), rhs->scores[n].size());
      lhs_counts.push_back(lhs->scores[n][p]);
      rhs_counts.push_back(rhs->scores[n][p]);
    }

    // zerograms
//    lhs_counts.push_back(762);
//    rhs_counts.push_back(762);
    std::vector<double> ppl_vec;
    for (size_t i=0;i<ppl_vec_.size();++i) ppl_vec.push_back(ppl_vec_[i]);
//    ppl_vec.push_back(::log(26.0));

    CHECK_EQ(lhs->scores.size(), rhs->scores.size());
    CHECK_EQ(lhs_counts.size(), rhs_counts.size());

    for (size_t i=0;i<lhs_counts.size();++i) {
      for (size_t j=i+1;j<lhs_counts.size();++j) {
        lhs_counts[j] -= lhs_counts[i];
        rhs_counts[j] -= rhs_counts[i];
      }
    }

    double lhs_ppl = 1.0;
    double rhs_ppl = 1.0;

    for (size_t i=0;i<lhs_counts.size();++i) {
      lhs_ppl += ppl_vec[i]*lhs_counts[i];
      rhs_ppl += ppl_vec[i]*rhs_counts[i];
    }

    return lhs_ppl > rhs_ppl;

  }
};

class ExtensionOrder {
public:
    ExtensionOrder(Vocab& f_vocab, std::vector<std::string>& special_words);
    //TextInputStream
    virtual ~ExtensionOrder();
    word_id_t getWordAt(word_id_t pos) const;
    word_id_t size() const;
    void fillHighestNgramFreqOrderBeam(std::vector<Counts*>& base_counts, std::vector<double> ppl_vec, size_t beam_size);
    void fillFromFile(Vocab& f_vocab, const std::string &fn);
    void fillExtOrder(Vocab& f_vocab, Embedding* embedding, size_t beam_size);
    void fillEmbedding(Vocab& f_vocab, Embedding* embedding, size_t beam_size);
    void writeToFile(const Vocab& f_vocab, const std::vector<Counts*> &base_counts, const std::string &fn) const;

private:
    bool compareGreater(std::vector<size_t>& big, std::vector<size_t>& small);
    count_t calcNumberOfAdditionalNgrams(const Counts& base_counts, const std::vector<bool>& wordsCalculated, word_id_t newWord, bool unique=false) const;

    std::vector<std::string>& special_words;
    std::vector<word_id_t> ext_order;
    std::vector<bool> words_calculated;
    std::set<word_id_t> words_to_calculate;
    std::vector<double> ppl_vec_;
};
#endif /* EXTENSION_ORDER_HH_ */
