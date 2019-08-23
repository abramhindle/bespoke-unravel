#include "ngram_feature.hh"

void NgramFeature::activate(const Counts& f_counts, KenLmWithVocab& e_lm,
                            order_t order, const ExtensionOrder& extensionOrder,
                            score_t initial_weight) {
  f_counts_ = &f_counts;
  e_lm_ = &e_lm;
  order_ = order;
  fillWeights(extensionOrder, initial_weight);
}

void NgramFeature::updateScore(const Mapping& hypothesis, word_id_t next_f,
                               score_t& old_score) const {
  score_t score = 0;
  // iterate over new ngrams
  if (next_f >= f_counts_->getNewNgrams().size()) {
    LOG(FATAL) << "FATAL: NO NEW NGRAMS FOR " << next_f
              << " - ONLY HAVE NGRAMS UP TO "
              << f_counts_->getNewNgrams().size() << std::endl;
  }
  CHECK_LT(next_f, f_counts_->getNewNgrams().size());
  for (std::set<Ngram*>::iterator it =
           f_counts_->getNewNgrams()[next_f].begin();
       it != f_counts_->getNewNgrams()[next_f].end(); ++it) {
    word_id_t plainNgram[order_];
    bool fullNgram = true;
    for (size_t i = 0; i < order_; ++i) {
      if (!hypothesis.hasF((*it)->t[i])) {
        fullNgram = false;
        break;
      } else {
        plainNgram[i] = hypothesis.getE((*it)->t[i]);
      }
    }
    if (fullNgram) {
      score += score_t((*it)->count) *
               score_t(e_lm_->ngramScore(&(plainNgram[0]), order_));
    }
  }

  if (normalize_) score /= score_t(f_counts_->getZerogramCount());
  old_score += score;
}

void NgramFeature::fillWeights(const ExtensionOrder& ext_order,
                               score_t initial_weight) {
  weights.resize(f_counts_->getVocabConst().size());
  for (word_id_t i = 0; i < ext_order.size(); ++i) {
    word_id_t newWord = ext_order.getWordAt(i);
    weights[newWord] = initial_weight;
  }
}
