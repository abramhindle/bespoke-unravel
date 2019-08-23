#ifndef ORACLE_FEATURE_H_
#define ORACLE_FEATURE_H_

#include <bitset>

#include "global.hh"
#include "feature.hh"
#include "mapping.hh"
#include "reference_mapping.hh"
#include "kenlm_with_vocab.hh"
#include "counts.hh"
#include "extension_order.hh"

class OracleFeature : public Feature {

public:
    OracleFeature(ReferenceMapping& reference_mapping, size_t len, score_t weight) : referenceMapping_(reference_mapping) {
        fillWeights(weight, len);
    };

    virtual ~OracleFeature() {
    }

    virtual void updateScore(const Mapping& hypothesis, word_id_t next_f, score_t& old_score) const;

private:
    const ReferenceMapping & referenceMapping_;
    void fillWeights(score_t weight, size_t len);

};

#endif /* ORACLE_FEATURE_H_ */
