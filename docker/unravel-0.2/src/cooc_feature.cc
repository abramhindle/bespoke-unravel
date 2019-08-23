#include "cooc_feature.hh"

void CoocFeature::activate(const ExtensionOrder& extensionOrder,
                           score_t initial_weight) {
  fillWeights(extensionOrder, initial_weight);
}

void CoocFeature::fillWeights(const ExtensionOrder& ext_order,
                              score_t initial_weight) {
  weights.resize(f_cooc_.getVocabConst().size());
  for (word_id_t i = 0; i < ext_order.size(); ++i) {
    word_id_t newWord = ext_order.getWordAt(i);
    weights[newWord] = initial_weight;
  }
}

void CoocFeature::updateScore(const Mapping& hypothesis, word_id_t next_f,
                              score_t& old_score) const {

  word_id_t next_e = hypothesis.getE(next_f);

  const std::set<word_id_t>* f_coocs = f_cooc_.getCooc(next_f);
  const std::set<word_id_t>* e_coocs = e_cooc_.getCooc(next_e);

  CHECK_NOTNULL(f_coocs);
  CHECK_NOTNULL(e_coocs);

  // then only words are candidates that do cooccur with all translated coocs
  for (std::set<word_id_t>::const_iterator it = f_coocs->begin();
       it != f_coocs->end(); ++it) {
    word_id_t f_prime = *it;
    word_id_t e_prime = hypothesis.getE(f_prime);

    if (e_prime == VOCAB_WORD_NOT_FOUND || e_prime == MAPPING_NOT_MAPPED)
      continue;

    if (e_coocs->count(e_prime) == 0) {
      if (verbose_) {
        LOG(INFO) << "REJECT because F COOC "
                  << f_cooc_.getVocabConst().getWord(f_prime) << " maps to "
                  << e_cooc_.getVocabConst().getWord(e_prime);
        LOG(INFO) << " which is not a COOC of "
                  << e_cooc_.getVocabConst().getWord(next_e) << std::endl;
        LOG(INFO) << std::endl;
        LOG(INFO) << "reject" << std::endl;
      }
      old_score -= 1.0;
    }
  }

  // keep old_score as it was
}
