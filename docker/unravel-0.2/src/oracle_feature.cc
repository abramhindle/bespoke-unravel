#include "oracle_feature.hh"

void OracleFeature::updateScore(const Mapping& hypothesis, word_id_t next_f,
                                score_t& old_score) const {
  // if wrong, then wrong forever
  if (old_score == -1.0) return;

  if (referenceMapping_.isSpecified(next_f)) {
    // as long as correct, set 
    if (referenceMapping_.isCorrect(hypothesis.getE(next_f), next_f)) {
      old_score = 1.0;
    } else {
      old_score = -1.0;
    }
  } else {
    // keep in last state
  }
}

void OracleFeature::fillWeights(score_t weight, size_t len) {
  weights.resize(len);
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = weight;
}
