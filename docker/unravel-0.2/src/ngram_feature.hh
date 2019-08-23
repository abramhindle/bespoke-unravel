#ifndef NGRAM_FEATURE_H_
#define NGRAM_FEATURE_H_

#include <bitset>

#include "feature.hh"
#include "mapping.hh"
#include "kenlm_with_vocab.hh"
#include "counts.hh"
#include "extension_order.hh"

class NgramFeature : public Feature {

public:
    NgramFeature(bool normalize) : normalize_(normalize) {
    };
    virtual ~NgramFeature() {
    }

    void activate(const Counts& f_counts, KenLmWithVocab &e_lm, order_t order, const ExtensionOrder& extensionOrder, score_t initial_weight);

    virtual void updateScore(const Mapping& hypothesis, word_id_t next_f, score_t& old_score) const;

    KenLmWithVocab& getELM() {
        return *e_lm_;
    }

private:
    const Counts* f_counts_;
    bool normalize_;
    KenLmWithVocab* e_lm_;
    order_t order_;
    void fillWeights(const ExtensionOrder& ext_order, score_t initial_weight);

};

#endif /* NGRAM_FEATURE_H_ */
