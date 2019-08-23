#include "embedding_feature.hh"
#include "misc.hh"

void EmbeddingFeature::activate() {
}

void EmbeddingFeature::updateScore(const Mapping& hypothesis, word_id_t next_f,
                               score_t& old_score) const {
  score_t score = 0.0;

  // need access to extension order
  word_id_t next_e = hypothesis.getE(next_f);

  if (e_embedding_.hasVec(next_e) ^ f_embedding_.hasVec(next_f)) {
    old_score -= 10.0; //penalty
    return;
  }
  if (!e_embedding_.hasVec(next_e) || !f_embedding_.hasVec(next_f)) return;

  CHECK(e_embedding_.hasVec(next_e));
  CHECK(f_embedding_.hasVec(next_f));

  auto& next_e_vec = e_embedding_.getVec(next_e);
  auto& next_f_vec = f_embedding_.getVec(next_f);

  double min_dist = 99.0;
  double max_dist = -99.0;

  const auto& next_e_closest = e_embedding_.getClosest(next_e);
  const auto& next_f_closest = f_embedding_.getClosest(next_f);

  for (size_t i=2; i<hypothesis.getCardinality(); ++i) {
    word_id_t cur_f = ext_order_.getWordAt(i);

    if (cur_f == next_f || !f_embedding_.getVocabConst().isNormal(cur_f) || !f_embedding_.hasVec(cur_f)) continue;

    word_id_t cur_e = hypothesis.getE(cur_f);
    if (!e_embedding_.getVocabConst().isNormal(cur_e)) continue;

    if (!e_embedding_.hasVec(cur_e) || !f_embedding_.hasVec(cur_f)) {
      continue;
    }

    CHECK(e_embedding_.hasVec(cur_e));
    CHECK(f_embedding_.hasVec(cur_f));

    auto& cur_e_vec = e_embedding_.getVec(cur_e);
    auto& cur_f_vec = f_embedding_.getVec(cur_f);

    const auto& cur_e_closest = e_embedding_.getClosest(cur_e);
    const auto& cur_f_closest = f_embedding_.getClosest(cur_f);

    // find rank of cur_e on next_e
    int cur_e_rank = 0;
    for (size_t i=0;i<next_e_closest.size();++i) {
      if (next_e_closest[i].first == cur_e) {
        cur_e_rank = i;
        break;
      }
    }

    // find rank of cur_f on next_f
    int cur_f_rank = 0;
    for (size_t i=0;i<next_f_closest.size();++i) {
      if (next_f_closest[i].first == cur_f) {
        cur_f_rank = i;
        break;
      }
    }

    size_t rank_diff_max = next_f_closest.size();
    int rank_diff = 0;
    if (cur_e_rank > cur_f_rank) {
      rank_diff = cur_e_rank - cur_f_rank;
    } else {
      rank_diff = cur_f_rank - cur_e_rank;
    }

    int max_rank = cur_e_rank;
    if (cur_f_rank > max_rank) max_rank = cur_f_rank;

    double reward = 1 + (static_cast<double>(rank_diff_max) - static_cast<double>(rank_diff)) / static_cast<double>(max_rank);

    if (cur_f_rank != 0 && cur_e_rank != 0) {
      score += static_cast<double>(reward);
    }

    double e_distance = Embedding::diff(next_e_vec, cur_e_vec);
    double f_distance = Embedding::diff(next_f_vec, cur_f_vec);

    if (e_distance < min_dist) min_dist = e_distance;
    if (e_distance > max_dist) max_dist = e_distance;

    // if they are FAR apart, then the exact distance is not interesting
    if (e_distance > max_distance_ && f_distance > max_distance_) continue;


    //LOG(INFO) << "difference vector f=(" << cur_f << ", " << next_f << ") = " << f_distance << " vs "
    //          << "(" << cur_e << ", " << next_e << ") = " << e_distance << endl;

//    double dist = ::log(1.0+(e_distance - f_distance) * (e_distance - f_distance));
      //dist = sqrt(dist);
     
//    score += -dist*0.02;
  }
//  LOG(INFO) << "min e_distance = " << min_dist << std::endl;
//  LOG(INFO) << "max e_distance = " << max_dist << std::endl;
  old_score += score;
}

void EmbeddingFeature::fillWeights(score_t weight) {
  weights.resize(f_embedding_.getVocabConst().size());
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = weight;
}
