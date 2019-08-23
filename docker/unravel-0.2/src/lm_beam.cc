// Copyright 2013 Malte Nuhn
// SOME BUG WHEN LM_BEAM=1

#include <fst/fstlib.h>
#include <fst/vector-fst.h>
#include <fst/shortest-path.h>
#include <fst/symbol-table.h>
#include <fst/fst.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/algorithm/string.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <limits>

#include "vocab.hh"
#include "mapping.hh"
#include "kenlm_with_vocab.hh"
#include "misc.hh"
#include "fst_permute.hh"
#include "fst_beamsearch.hh"
#include "lm_beam.hh"

score_t LmBeam::score_primary(const KenLmWithVocab &lm,
                              const lm::ngram::State &state, word_id_t word) {
  lm::ngram::State state2;
  return -lm.getModel().Score(state, word, state2);
}

// find best extensions given state and beamsize
std::vector<std::pair<word_id_t, score_t> > LmBeam::get_best(
    const KenLmWithVocab &lm, const lm::ngram::State &state,
    size_t lm_beam_size) {
  std::vector<std::pair<word_id_t, score_t> > scores, res;

  for (word_id_t j = 1; j < eVocab_.size(); ++j) {
    score_t score = score_primary(lm, state, j);
    scores.push_back(std::pair<word_id_t, score_t>(j, score));
  }
  size_t min = std::min(scores.size(), lm_beam_size);
  std::partial_sort(scores.begin(), scores.begin() + min, scores.end(),
                    misc::compareWordScorePair);
  for (word_id_t i = 0; i < min; ++i) {
    word_id_t e = scores[i].first;
    score_t score = scores[i].second;
    CHECK_NE(e, 0);
    CHECK_LT(e, eVocab_.size());
    res.push_back({e, score});
  }
  return res;
}

void LmBeam::init(const KenLmWithVocab &lm,
                  const std::vector<std::vector<word_id_t> > &states) {
  for (size_t i = 0; i < states.size(); ++i) {
    misc::update(0.2, i == 0) << i << "/" << states.size() << " ["
                              << 100 * double(i) / double(states.size())
                              << "%] states prepared"
                              << "\r";
    // loop over all substrings
    for (size_t a = 0; a < states[i].size(); ++a) {
      for (size_t b = a; b < states[i].size(); ++b) {
        // get str
        std::vector<word_id_t> sub(states[i].begin() + a,
                                   states[i].begin() + b + 1);
        // get state
        lm::ngram::State lm_state = lm.getNgramState(sub);
        // maybe we already covered this state
        if (beam_map_.find(lm_state) != beam_map_.end()) continue;
        // get best succesors
        const std::vector<std::pair<word_id_t, score_t> > &best =
            get_best(lm, lm_state, beamSize_);
        // append
        beam_map_[lm_state]
            .insert(beam_map_[lm_state].end(), best.begin(), best.end());
      }
    }
  }
  cerr << endl;
  LOG(INFO) << "did prepare " << beam_map_.size() << " states" << endl;
}

// get best lm extension
void LmBeam::addCandidates(word_id_t f, lm::ngram::State state,
                           std::set<word_id_t> *candidates) const {

  CHECK_NOTNULL(candidates);
  if (f == fVocab_.sos() || f == fVocab_.eos()) return;

  std::map<lm::ngram::State,
           std::vector<std::pair<word_id_t, score_t> > >::const_iterator it =
      beam_map_.find(state);
  CHECK(it != beam_map_.end());

  size_t added = 0;

  // add beam_map_ entries
  const std::vector<std::pair<word_id_t, score_t> > &lm_beam_to_expand =
      (*it).second;

  for (size_t j = 0; j < lm_beam_to_expand.size(); ++j) {
    word_id_t e = lm_beam_to_expand[j].first;
    CHECK(!misc::any_of(e, {eVocab_.null(), eVocab_.unk()}));
    // don't add "<s>" or "</s>" if we are in the middle of the sentence
    if (eVocab_.isNormal(e) == fVocab_.isNormal(f)) {
      CHECK(!((f == fVocab_.eos()) && (e == eVocab_.null())));
      CHECK_NE(f, fVocab_.null());
      CHECK_NE(e, eVocab_.null());
      candidates->insert(e);
      ++added;
    }
  }

  // CHECK_GT(added, 0);
}
