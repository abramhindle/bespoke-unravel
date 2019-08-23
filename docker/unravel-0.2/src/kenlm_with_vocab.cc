#include "kenlm_with_vocab.hh"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "actual_strings_kenlm.hh"
#include <queue>

using std::cerr;
using std::endl;


KenLmWithVocab::KenLmWithVocab()
    : model(0) {
}

KenLmWithVocab::KenLmWithVocab(const std::string &file, Vocab* vocab) {
  read(file, vocab);
}

KenLmWithVocab::~KenLmWithVocab() {
  if (model != 0) delete model;
}

void KenLmWithVocab::read(const std::string &file, Vocab* vocab) {
  LOG(INFO) << "reading lm '" << file << "'";
  // mainly this is all about generating an vocab object with the right content
  this->vocab_ = vocab;
  lm::ngram::Config lmConfig;
  ActualStringsKenLm actualStrings(*vocab_);
  lmConfig.enumerate_vocab = &actualStrings;

  model = new lm::ngram::Model(file.c_str(), lmConfig);
  LOG(INFO) << "lm loaded";
  LOG(INFO) << "lm vocab size: " << vocab_->size();
  first_invalid_vocab_id_ = vocab_->size();
}


lm::ngram::Model const& KenLmWithVocab::getModel() const {
  return *model;
}


bool KenLmWithVocab::validWordId(word_id_t word_id) const {
  return word_id < first_invalid_vocab_id_;
}

size_t KenLmWithVocab::getFirstInvalidVocabId() {
  return first_invalid_vocab_id_;
}


KenLmWithVocab::state_t KenLmWithVocab::getNullContextState() const {
  return model->NullContextState();
}

KenLmWithVocab::state_t KenLmWithVocab::getState(
    const KenLmWithVocab::state_t& state, word_id_t word) const {
  KenLmWithVocab::state_t state_new;
  model->Score(state, word, state_new);
  return state_new;
}


score_t KenLmWithVocab::getUnigramScore(word_id_t word) const {
  KenLmWithVocab::state_t dummy_state;
  return model->Score(model->NullContextState(), word, dummy_state);
}


// reverse
KenLmWithVocab::state_t KenLmWithVocab::getNgramState(
    const std::vector<word_id_t>& ngram) const {
  KenLmWithVocab::state_t state = model->NullContextState();
  KenLmWithVocab::state_t state_new = state;
  for (size_t i = 0; i < ngram.size(); ++i) {
    model->Score(state, ngram[ngram.size() - 1 - i], state_new);
    state = state_new;
  }
  return state;
}


std::string KenLmWithVocab::stateToString(const lm::ngram::State &state) const {
  std::stringstream sstr;
  sstr << "State:";
  for (size_t i=0;i<state.length;++i) {
    sstr << " " << vocab_->getWord(state.words[i]);
  }
  return sstr.str();
}


std::string KenLmWithVocab::stateToStringNormalOrder(const lm::ngram::State &state) const {
  std::stringstream sstr;
  size_t to_fill = getModel().Order() - state.length - 1;
  for (size_t i = 0; i < to_fill; ++i) {
    sstr << " #";
  }
  for (size_t i = state.length; i > 0; --i) {
    sstr << " " << vocab_->getWord(state.words[i-1]);
  }
  return sstr.str();
}


void KenLmWithVocab::addCandidates(const lm::ngram::State &state, std::set<word_id_t> *candidates, size_t max_candidates) const {

  CHECK_NOTNULL(candidates);
  const auto &it = best_successors_.find(state);
  CHECK(it != best_successors_.end()) << "no known successors for state " << stateToString(state);

  //size_t added = 0;

  // add beam_map_ entries
  const auto &lm_beam_to_expand = (*it).second;

  //LOG(INFO) << "added " << added << " lm beam candidates for state '" << stateToString(state) << "'";
  for (size_t j = 0; j < lm_beam_to_expand.size(); ++j) {
    if (j > 0 && j > max_candidates) break;
    word_id_t e = lm_beam_to_expand[j];
    CHECK_LT(e, first_invalid_vocab_id_);
    //LOG(INFO) << "with state '" << stateToString(state) << "' adding " << vocab_->getWord(e);
    CHECK(vocab_->isNormal(e));
    candidates->insert(e);
    //++added;
  }
}

void KenLmWithVocab::prepare_known_states() {
  CHECK_NOTNULL(vocab_);
  CHECK_NOTNULL(model);

  std::set<KenLmWithVocab::state_t> states_new;
  known_states_.insert(model->NullContextState());

  LOG(INFO) << "obtaining all LM states";
  while(known_states_.size() != states_new.size()) {
    known_states_.insert(states_new.begin(), states_new.end());
    LOG(INFO) << "number of states = " << known_states_.size();
    for (const KenLmWithVocab::state_t &cur_state : known_states_) {
      states_new.insert(cur_state);

      for (size_t word = 0; word < vocab_->size(); ++word) {
        KenLmWithVocab::state_t new_state = getState(cur_state, word);
        if (!states_new.count(new_state)) {
          states_new.insert(new_state);
        }
      }
    }
  }
}

void KenLmWithVocab::prepare_successors(size_t beam_size) {

  size_t states_prepared = 0;
  size_t successors_added = 0;

  state_t dummy_state;

  // get best scoring successors
  for (const KenLmWithVocab::state_t &cur_state : known_states_) {
    ++states_prepared;

    std::priority_queue<std::pair<score_t, word_id_t>> best_candidates;
    for (size_t word = 0; word < vocab_->size(); ++word) {
      score_t score = getScore(cur_state, word, &dummy_state);
      // minus needed for correct sorting order
      best_candidates.push(std::make_pair(-score, word));
      // pop worst scoring elements
      if (best_candidates.size() > beam_size) {
        best_candidates.pop();
      }
    }

    auto &cur_best_successors = best_successors_[cur_state];

    cur_best_successors = {};

    while(best_candidates.size() > 0) {
      auto word = best_candidates.top().second;
      if (vocab_->isNormal(word)) {
        cur_best_successors.push_back(word);
        ++successors_added;
      }
      best_candidates.pop();
    }

    // best successors should come first
    std::reverse(cur_best_successors.begin(), cur_best_successors.end());
  }

  LOG(INFO) << "prepared lookup for " << states_prepared << " states";
  LOG(INFO) << "added " << successors_added << " successors";
}
