#include "multi_ngram_feature.hh"

/*
 * TODO:
 * TODO: WORK IN PROGRESS !!!!!
 * TODO:
 */

void MultiNgramFeature::addCounts(const Counts& f_counts) {
  /* ensure the correct order of the counts */
  if (f_countss_.empty()) {
    max_order_ = f_counts.getOrder();
  } else {
    CHECK_EQ(f_countss_.back()->getOrder(), (f_counts.getOrder() + 1));
  }
  f_countss_.push_back(&f_counts);
}

void MultiNgramFeature::activate(KenLmWithVocab& e_lm,
                                 const ExtensionOrder& extensionOrder,
                                 score_t initial_weight) {
  e_lm_ = &e_lm;
  fillWeights(extensionOrder, initial_weight);
}

void MultiNgramFeature::updateScore(const Mapping& hypothesis, word_id_t next_f,
                                    score_t& old_score) const {
  score_t score = 0;
  // iterate over new ngrams
  for (size_t i = 0; i < f_countss_.size(); ++i) {
    const Counts* f_counts_ = f_countss_[i];
    const order_t order = f_counts_->getOrder();
    bool not_last_counts = (i != f_countss_.size() - 1);
    if (next_f >= f_counts_->getNewNgrams().size()) {
      std::cerr << "FATAL: NO NEW NGRAMS FOR " << next_f
                << " - ONLY HAVE NGRAMS UP TO "
                << f_counts_->getNewNgrams().size() << std::endl;
    }
    CHECK_LT(next_f, f_counts_->getNewNgrams().size());
    for (std::set<Ngram*>::iterator it =
             f_counts_->getNewNgrams()[next_f].begin();
         it != f_counts_->getNewNgrams()[next_f].end(); ++it) {
      word_id_t plainNgram[order];
      bool fullNgram = true;
      for (size_t i = 0; i < order; ++i) {
        if (!hypothesis.hasF((*it)->t[i])) {
          fullNgram = false;
          break;
        } else {
          plainNgram[i] = hypothesis.getE((*it)->t[i]);
        }
      }
      if (fullNgram) {
        score += score_t((*it)->count) *
                 score_t(e_lm_->ngramScore(&(plainNgram[0]), order));
        if (not_last_counts) {
          score -= score_t((*it)->count) *
                   score_t(e_lm_->ngramScore(&(plainNgram[1]), order - 1));
        }
      }
    }
  }
  if (normalize_) score /= score_t(f_countss_.back()->getZerogramCount());
  old_score += score;
}

void MultiNgramFeature::fillWeights(const ExtensionOrder& ext_order,
                                    score_t initial_weight) {
  CHECK_GE(f_countss_.size(), 1);
  LOG(INFO) << "MultiNgramFeature.weights.size = " << f_countss_.back()->getVocabConst().size();
  weights.resize(f_countss_.back()->getVocabConst().size());
  for (word_id_t i = 0; i < ext_order.size(); ++i) {
    word_id_t newWord = ext_order.getWordAt(i);
    weights[newWord] = initial_weight;
  }
}
