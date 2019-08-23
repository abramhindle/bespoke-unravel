#ifndef FEATURE_H_
#define FEATURE_H_

#include <bitset>

#include "global.hh"
#include "mapping.hh"
#include "cooc.hh"
#include "kenlm_with_vocab.hh"
#include "counts.hh"
#include "extension_order.hh"

class Feature {

public:
    Feature() : max_card(MAX_LEN+1) {
    }
    virtual ~Feature() {
    }
    virtual void updateScore(const Mapping& hypothesis, word_id_t next_f, score_t& old_score) const = 0;
    bool active(size_t current_card) {
        return (current_card <= max_card);
    }
    score_t getWeight(word_id_t next_f) {
        CHECK_LT(next_f, weights.size());
        return weights[next_f];
    }
    void setMaxCard(size_t max_card) {
        this->max_card = max_card;
    }
protected:
    // from word index to weight
    std::vector<score_t> weights;
    size_t max_card;
};
#endif /* FEATURE_H_ */
