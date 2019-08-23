#include "phrase_feature.hh"

void PhraseFeature::updateScore(const Mapping& hypothesis, word_id_t next_f,
                                score_t& old_score) const {
  old_score += 0.0;
  order_t order = fCounts_.getOrder();
  // NEED FAST CODE TO LOOKUP COUNT OF A PHRASE AKA GIVEN NGRAM
  // NEED FIXED NGRAMS FOR CURRENT F
  for (std::set<Ngram*>::iterator it = fCounts_.getNewNgrams()[next_f].begin();
       it != fCounts_.getNewNgrams()[next_f].end(); ++it) {
    Ngram& cipher = **it;
    Ngram plain(order);
    bool full = hypothesis.mapNgram_(cipher.t, plain.t, order);
    if (full) {

      score_t cnt = 0;
      std::map<Ngram, score_t>::const_iterator it = count_map_.find(plain);
      if (it != count_map_.end()) cnt = it->second;
      score_t logprob = ::log(cnt);
      if (logprob < -99) logprob = -99.0;
      old_score += cipher.count * logprob;
      // if (cipher.count>1000) {
      //    for (size_t i=0;i<order;++i) std::cerr <<
      // fCounts_.getVocabConst().getWord(cipher.t[i]) << " "; std::cerr <<
      // cipher.count << " -- ";
      //    for (size_t i=0;i<order;++i) std::cerr <<
      // eCounts_.getVocabConst().getWord(plain.t[i]) << " "; std::cerr << cnt
      // << std::endl;
      //}
    }
  }
}

void PhraseFeature::fillWeights(score_t weight) {
  weights.resize(fCounts_.getVocab().size());
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = weight;
}
