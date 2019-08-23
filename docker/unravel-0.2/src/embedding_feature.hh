#ifndef EMBEDDING_FEATURE_H_
#define EMBEDDING_FEATURE_H_

#include <bitset>

#include "feature.hh"
#include "mapping.hh"
#include "embedding.hh"
#include "kenlm_with_vocab.hh"
#include "counts.hh"
#include "extension_order.hh"

class EmbeddingFeature : public Feature {

public:
    EmbeddingFeature(const Embedding& eEmbedding, const Embedding& fEmbedding, const ExtensionOrder& ext_order, score_t weight, double max_distance) : e_embedding_(eEmbedding), f_embedding_(fEmbedding), ext_order_(ext_order), max_distance_(max_distance) {
        fillWeights(weight);
    };
    virtual ~EmbeddingFeature() {
    }

    void activate();

    virtual void updateScore(const Mapping& hypothesis, word_id_t next_f, score_t& old_score) const;

private:

    void fillWeights(score_t initial_weight);
    const Embedding& e_embedding_, f_embedding_;
    const ExtensionOrder& ext_order_;
    double max_distance_;
};

#endif /* EMBEDDING_FEATURE_H_ */
