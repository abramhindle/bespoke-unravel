#ifndef COOC_FEATURE_H_
#define COOC_FEATURE_H_

#include <bitset>

#include "global.hh"
#include "feature.hh"
#include "mapping.hh"
#include "reference_mapping.hh"
#include "kenlm_with_vocab.hh"
#include "counts.hh"
#include "extension_order.hh"

class CoocFeature : public Feature {
 public:
  CoocFeature(const Cooc& e, const Cooc& f, bool verbose = false)
      : e_cooc_(e), f_cooc_(f), verbose_(verbose) {
    std::cerr << "CoocConstraint: setting up vectors" << std::endl;
    std::cerr << "CoocConstraint: filling vectors" << std::endl;
  }
  virtual ~CoocFeature() {}
  void activate(const ExtensionOrder& extensionOrder, score_t initial_weight);
  void fillWeights(const ExtensionOrder& ext_order, score_t initial_weight);
  virtual void updateScore(const Mapping& hypothesis, word_id_t next_f,
                           score_t& old_score) const;

 private:
  const Cooc& e_cooc_;
  const Cooc& f_cooc_;
  const bool verbose_;
};

#endif /* ORACLE_FEATURE_H_ */
