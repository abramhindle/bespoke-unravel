#ifndef PHRASE_FEATURE_H_
#define PHRASE_FEATURE_H_

#include <bitset>

#include "global.hh"
#include "feature.hh"
#include "mapping.hh"
#include "kenlm_with_vocab.hh"
#include "counts.hh"
#include "ngram.hh"
#include "extension_order.hh"

class PhraseFeature : public Feature {

public:
    PhraseFeature(Counts& e, Counts& f, score_t weight) : eCounts_(e), fCounts_(f) {
        fillWeights(weight);

        std::cerr << "setting up lookup map" << std::endl;
        const std::vector<Ngram> & vec = eCounts_.getNgrams();
        for (size_t i=0;i<vec.size();++i) count_map_[vec[i]] = score_t(vec[i].count)/score_t(eCounts_.getZerogramCount());

    };

    virtual ~PhraseFeature() {
    }

    virtual void updateScore(const Mapping& hypothesis, word_id_t next_f, score_t& old_score) const;

private:
    void fillWeights(score_t weight);
    Counts &eCounts_;
    Counts &fCounts_;
    std::map<Ngram, score_t> count_map_;
};

#endif /* PHRASE_FEATURE_H_ */
