#ifndef MULTI_NGRAM_FEATURE_HH_
#define MULTI_NGRAM_FEATURE_HH_

#include "feature.hh"

class MultiNgramFeature: public Feature {
public:
    MultiNgramFeature(bool normalize) : normalize_(normalize) {
    };
    virtual ~MultiNgramFeature() {
    }

    void addCounts(const Counts& f_counts);
    void activate(KenLmWithVocab &e_lm, const ExtensionOrder& extensionOrder, score_t initial_weight);

    virtual void updateScore(const Mapping& hypothesis, word_id_t next_f, score_t& old_score) const;

    KenLmWithVocab& getELM() {
        return *e_lm_;
    }

private:
    std::vector<const Counts*> f_countss_;
    bool normalize_;
    KenLmWithVocab* e_lm_;
    order_t max_order_;
    void fillWeights(const ExtensionOrder& ext_order, score_t initial_weight);
};

#endif /* MULTI_NGRAM_FEATURE_HH_ */
