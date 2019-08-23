#include "automaton_lm.hh"

#include <random>


AutomatonLM::AutomatonLM(KenLmWithVocab& klm) {
  vocab_ = &(klm.getVocabNonConst());
}

AutomatonLM::~AutomatonLM() {
}

score_t AutomatonLM::getScore(state_t state, word_id_t word,
                 state_t* out_state) const {
  *out_state = automaton_[state][word].state;
  return automaton_[state][word].score;
}

size_t AutomatonLM::getSize() const {
  return automaton_.size();
}

AutomatonLM::state_t AutomatonLM::getNullContextState() const {
  return 0;
}

void AutomatonLM::build_automaton_lm(const KenLmWithVocab& klm) {
  misc::ProcessStopWatch sw;
  size_t state_count = 0;
  automaton_.clear();
  std::map<lm::ngram::State, size_t> known;
  std::map<lm::ngram::State, size_t>* toExpand_source
    = new std::map<lm::ngram::State, size_t>();
  std::map<lm::ngram::State, size_t>* toExpand_target
    = new std::map<lm::ngram::State, size_t>();
  known[klm.getModel().NullContextState()] = 0;
  (*toExpand_source)[klm.getModel().NullContextState()] = 0;
  ++state_count;

  while (!toExpand_source->empty()) {
    misc::ProcessStopWatch sw_expansion;
    automaton_.resize(automaton_.size() + toExpand_source->size(),
                      std::vector<StateAndScore>(klm.getVocabConst().size()));
    for (auto state_and_index : *toExpand_source) {
      const lm::ngram::State& state = state_and_index.first;
      size_t index = state_and_index.second;
      for (size_t e = 0; e < klm.getVocabConst().size(); ++e) {
        lm::ngram::State out_state;
        score_t score = klm.getModel().Score(state, e, out_state);
        auto out_state_and_index = known.find(out_state);
        if (out_state_and_index == known.end()) {
          size_t out_index = state_count;
          ++state_count;
          (*toExpand_target)[out_state] = out_index;
          known[out_state] = out_index;
          automaton_[index][e].score = score;
          automaton_[index][e].state = out_index;
        } else {
          size_t out_index = out_state_and_index->second;
          automaton_[index][e].score = score;
          automaton_[index][e].state = out_index;
        }
      }
    }
    sw_expansion.store();
    LOG(INFO) << "expanded " << toExpand_source->size() << " states to "
        << toExpand_target->size() << " new states in "
        << sw_expansion.wall_millis().count() << "ms";
    std::swap(toExpand_source, toExpand_target);
    toExpand_target->clear();
  }

  delete toExpand_source;
  delete toExpand_target;

  CHECK_EQ(automaton_.size(), state_count);

  sw.store();
  LOG(INFO) << "build automaton with " << state_count << " states in "
      << sw.wall_millis().count() << "ms";

  check_automaton_lm(klm, 100000);
}


void AutomatonLM::check_automaton_lm(const KenLmWithVocab& klm, size_t n_queries) {
  // [0.0 ... 1.0)
  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
  // also seeds engine
  std::default_random_engine random_engine(std::random_device{}());
  std::vector<word_id_t> rnd_e_seq(n_queries);
  for (size_t i = 0; i < n_queries; ++i) {
    double rand_num = uniform_dist(random_engine);
    rnd_e_seq[i] = (klm.getVocabConst().size()) * rand_num;
  }

  score_t total_score = 0;
  lm::ngram::State state = klm.getModel().NullContextState();
  lm::ngram::State out_state;
  state_t state_ = 0;
  state_t out_state_;
  misc::ProcessStopWatch sw;
  for (size_t i = 0; i < n_queries; ++i) {
    word_id_t e = rnd_e_seq[i];
    score_t score = klm.getScore(state, e, &out_state);
    state = out_state;
    score_t score_ = getScore(state_, e, &out_state_);
    state_ = out_state_;
    CHECK_EQ(score, score_);
    total_score += score_;
  }
  sw.store();
  LOG(INFO) << "checked correctness of automaton lm scores for "
      << n_queries << " queries in " << sw.wall_millis().count()
      << "ms. total score: " << total_score;
}
