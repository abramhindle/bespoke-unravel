#ifndef LM_BEAM_H_
#define LM_BEAM_H_

#include "global.hh"
#include "vocab.hh"
#include "kenlm_with_vocab.hh"
#include "misc.hh"
#include <vector>
#include <boost/iostreams/filtering_stream.hpp>
#include <set>

typedef std::map<lm::ngram::State, std::vector<std::pair<word_id_t,score_t> > > LmBeamMap;

class LmBeam {
public:
    
    void addCandidates(word_id_t f, lm::ngram::State state, std::set<word_id_t> * candidates) const;
    void init(const KenLmWithVocab &lm, const std::vector<std::vector<word_id_t> > & states);

    LmBeam(Vocab & eVocab, Vocab & fVocab, size_t beamSize) : eVocab_(eVocab), fVocab_(fVocab), beamSize_(beamSize) {

    };

    virtual ~LmBeam() {
    };

private:

    score_t score_primary(const KenLmWithVocab &lm, const lm::ngram::State & state, word_id_t word);
    std::vector<std::pair<word_id_t,score_t> > get_best(const KenLmWithVocab &lm, const lm::ngram::State & state, size_t lm_beam_size);
    Vocab & eVocab_, fVocab_;
    size_t beamSize_;
    LmBeamMap beam_map_;
};

#endif /* LM_BEAM_H_ */
