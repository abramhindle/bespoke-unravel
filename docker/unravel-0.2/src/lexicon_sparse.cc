#define LN_10 2.302585093

#include <glog/logging.h>
#include <gflags/gflags.h>

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
#include "lexicon_sparse.hh"

using std::endl;

const std::vector<word_id_t>& LexiconSparse::getBeam(word_id_t f) const {
  return lex_beam_[f];
}

size_t LexiconSparse::addCandidates(word_id_t f,
                                    std::set<word_id_t>* candidates,
                                    size_t limit) const {
  CHECK_NOTNULL(candidates);
  CHECK(f_vocab_->isNormal(f) || f == f_vocab_->null());
  CHECK_LT(f, lex_beam_.size());

  const std::vector<word_id_t>& cur_beam = lex_beam_[f];
  size_t added = 0;

  // add cur_beam entries
  for (size_t j = 0; j < std::min(cur_beam.size(), limit); ++j) {
    word_id_t e = cur_beam[j];
    CHECK(e_vocab_->isNormal(e) || e == e_vocab_->null());

    if (score(f, e) > -99) {
      candidates->insert(e);
      ++added;
    }
  }

  return added;
}

void LexiconSparse::init() {
  CHECK_NOTNULL(f_vocab_);
  CHECK_NOTNULL(e_vocab_);

  VLOG(1) << "initializing lexicon" << endl;
  // initialize
  ef_counts_.clear();
  ef_counts_.resize(e_vocab_->size());
  for (size_t i = 0; i < e_vocab_->size(); ++i) {
    ef_counts_[i].clear();
  }
}

score_t LexiconSparse::lin_score_fast(word_id_t f, word_id_t e) const {
  CHECK_LT(e, ef_counts_.size()) << e_vocab_->getWord(e);
  const auto& ef_count = ef_counts_[e].find(f);
  if (ef_count == ef_counts_[e].end()) {
    return 0.0;
  } else {
    return ef_count->second;
  }
}

score_t LexiconSparse::ibm1(const std::vector<word_id_t>& f_sentence,
                            const std::vector<word_id_t>& e_sentence) {

  score_t result = 0.0;

  // loop over input sentence
  for (size_t j = 0; j < f_sentence.size(); ++j) {
    word_id_t f_idx = f_sentence[j];
    double prob = 0.0;

    // loop over current e sentence
    for (size_t i = 0; i < e_sentence.size(); ++i) {
      word_id_t e_idx = e_sentence[i];
      double lex_score = score(f_idx, e_idx);
      prob += exp(lex_score);
    }

    prob /= e_sentence.size();
    result += -log(prob);
  }

  // score /= double(e_sentences_idx_[e].size());
  // score += 10*std::abs(double(e_sentences_idx_[e].size()) -
  // double(f_sentences_idx_[id].size()));

  return result;
}

std::map<size_t, size_t> LexiconSparse::ibm1Map(
    const std::vector<word_id_t>& f_sentence,
    const std::vector<word_id_t>& e_sentence, score_t threshold) {

  std::map<size_t, size_t> result;

  size_t holes = 0;
  size_t unaligned = 0;
  // loop over input sentence
  bool found_last = false;

  for (size_t j = 0; j < f_sentence.size(); ++j) {
    word_id_t f_idx = f_sentence[j];

    bool found = false;
    // loop over current e sentence
    for (size_t i = 0; i < e_sentence.size(); ++i) {
      word_id_t e_idx = e_sentence[i];
      double lex_score = score(f_idx, e_idx);
      double prob = exp(lex_score);
      if (prob > threshold) {
        result[i] = j;
        found = true;
        break;
      }
    }
    if (found_last && !found) ++holes;
    // early exit
    if (holes > 1) return result;
    found_last = found;
    if (!found) ++unaligned;
    // early exit
    if (unaligned > 3) return result;
  }
  return result;
}

// log[p(f|e)] -- only normal words and null are used for learning
score_t LexiconSparse::score(word_id_t f, word_id_t e) const {
  CHECK_NOTNULL(e_vocab_);
  CHECK_NOTNULL(f_vocab_);

  CHECK((e_vocab_->isNormal(e) || e_vocab_->isNull(e)) &&
        (f_vocab_->isNormal(f) || f_vocab_->isNull(f)))
      << "only normal words and null are allowed for learning, but queried "
         "with e=" << e_vocab_->getWord(e) << " f=" << f_vocab_->getWord(f);

  score_t n_f_words = static_cast<score_t>(f_vocab_->normal_plus_null_size());

  return log(delta_lambda_ * lin_score_fast(f, e) +
             (1 - delta_lambda_) / n_f_words);
}

// only normal words and null are used for learning
void LexiconSparse::add(word_id_t e, word_id_t f, score_t lin_score) {
  CHECK_NOTNULL(e_vocab_);
  CHECK_NOTNULL(f_vocab_);
  CHECK(e_vocab_->isNormal(e) || e_vocab_->isNull(e))
      << "e=" << e_vocab_->getWord(e);
  CHECK(f_vocab_->isNormal(f) || f_vocab_->isNull(f))
      << "f=" << f_vocab_->getWord(f);
  CHECK(std::isfinite(lin_score));
  ef_counts_[e][f] += lin_score;
}

// max_evocab_id is needed, because the vocabulary might contain tokens that
// are only available in the reference output.
//
// the vocabulary has to be setup in a way that all reference-only tokens
// are stored with id's bigger than emax_vocab_id, otherwise we are implictly
// cheating
void LexiconSparse::setupBeam(size_t max_evocab_id) {
  CHECK_NOTNULL(e_vocab_);
  CHECK_NOTNULL(f_vocab_);
  CHECK_LE(max_evocab_id, e_vocab_->size());

  LOG(INFO) << "setup_lex_beam with beamsize=" << beamSize_;
  lex_beam_.clear();
  lex_beam_.resize(f_vocab_->size());
  std::vector<word_id_t> f_normal_plus_null = f_vocab_->getNormalWords();
  f_normal_plus_null.push_back(f_vocab_->null());
  for (word_id_t f : f_normal_plus_null) {
    std::vector<std::pair<word_id_t, score_t> > scores;

    for (word_id_t e : e_vocab_->getNormalWords()) {
      if (e > max_evocab_id) break;
      score_t cur_score = -score(f, e);
      scores.push_back(std::pair<word_id_t, score_t>(e, cur_score));
    }

    size_t min = std::min(scores.size(), beamSize_);
    std::sort(scores.begin(), scores.end(), misc::compareWordScorePair);

    lex_beam_[f].clear();
    lex_beam_[f].push_back(e_vocab_->null());
    for (word_id_t i = 0; i < min; ++i) {
      if (scores[i].second >= 99) break;
      word_id_t e = scores[i].first;
      CHECK_LT(scores[i].second, 99);
      lex_beam_[f].push_back(e);
      VLOG(2) << "adding entry " << e_vocab_->getWord(e) << " " << f_vocab_->getWord(f) << " to lex beam (score=" << scores[i].second << ")";
    }
    if (lex_beam_[f].size() == 0) {
      LOG(WARNING) << "lex_beam for word f=" << f << " ("
                   << f_vocab_->getWord(f)
                   << ") is empty, because log(p)>99 for all e" << endl;
    }
  }
}

void LexiconSparse::learn() {
  CHECK_NOTNULL(e_vocab_);
  CHECK_NOTNULL(f_vocab_);

  LOG(INFO) << "learn normalize" << endl;
  for (size_t e = 0; e < e_vocab_->size(); ++e) {
    score_t sum = 0.0;
    for (auto it = ef_counts_[e].begin(); it != ef_counts_[e].end(); ++it) {
      sum += it->second;
    }
    for (auto it = ef_counts_[e].begin(); it != ef_counts_[e].end(); ++it) {
      it->second /= sum;
    }
  }

  LOG(INFO) << "done";
}

void LexiconSparse::learn_norm_reverse(const std::vector<score_t>& p_e,
                                       const std::vector<score_t>& p_f) {
  CHECK_NOTNULL(e_vocab_);
  CHECK_NOTNULL(f_vocab_);

  LOG(INFO) << "learn normalize reverse" << endl;

  std::vector<score_t> sum(f_vocab_->size(), 0.0);
  for (size_t e = 0; e < e_vocab_->size(); ++e) {
    for (auto it = ef_counts_[e].begin(); it != ef_counts_[e].end(); ++it) {
      sum[it->first] += it->second;
    }
  }
  for (size_t e = 0; e < e_vocab_->size(); ++e) {
    for (auto it = ef_counts_[e].begin(); it != ef_counts_[e].end(); ++it) {
      it->second *= (p_f[it->first] / p_e[e]) / sum[it->first];
    }
  }

  learn();

  LOG(INFO) << "done";
}

void LexiconSparse::constrain_on_e(const std::vector<word_id_t>& constrained_e,
                                   size_t n, score_t boost_factor) {
  for (word_id_t e : constrained_e) {
    std::vector<misc::WordScore> wsps;
    for (auto it = ef_counts_[e].begin(); it != ef_counts_[e].end(); ++it) {
      wsps.push_back(misc::WordScore(it->first, it->second));
    }
    std::sort(wsps.begin(), wsps.end(), misc::compareWordScorePairBigger);
    std::stringstream ss;
    ss << "e=" << e_vocab_->getWord(e) << " :";
    //    score_t sum = 0;
    size_t effective_n = std::min(wsps.size(), n);
    for (size_t i = 0; i < effective_n; ++i) {
      //      sum += wsps[i].second;
      ef_counts_[e][wsps[i].first] = wsps[i].second * boost_factor;
      ss << " " << f_vocab_->getWord(wsps[i].first) << "("
         << ef_counts_[e][wsps[i].first] << ")";
    }
    LOG(INFO) << ss.str();
  }
}

void LexiconSparse::constrain_on_f(const std::vector<word_id_t>& constrained_f,
                                   size_t m, score_t boost_factor) {
  for (word_id_t f : constrained_f) {
    std::vector<misc::WordScore> wsps;
    for (size_t e = 0; e < e_vocab_->size(); ++e) {
      auto it = ef_counts_[e].find(f);
      if (it != ef_counts_[e].end()) {
        wsps.push_back(misc::WordScore(e, it->second));
      }
    }
    std::sort(wsps.begin(), wsps.end(), misc::compareWordScorePairBigger);
    std::stringstream ss;
    ss << "f=" << f_vocab_->getWord(f) << " :";
    size_t effective_m = std::min(wsps.size(), m);
    for (size_t i = 0; i < effective_m; ++i) {
      ef_counts_[wsps[i].first][f] = wsps[i].second * boost_factor;
      ss << " " << e_vocab_->getWord(wsps[i].first) << "("
         << ef_counts_[wsps[i].first][f] << ")";
    }
    LOG(INFO) << ss.str();
  }
}

void LexiconSparse::printLex(score_t thresh) const {
  std::cout << "printing smoothed probability p(f|e) [interpolation_lambda="
            << delta_lambda_ << "] for every e one line" << endl;
  for (word_id_t e : e_vocab_->getNormalWords()) {
    std::cout << e_vocab_->getWord(e) << "\t";
    score_t sum = 0.0;
    for (word_id_t f : f_vocab_->getNormalWords()) {
      score_t prob = exp(score(f, e));
      if (prob >= thresh) {
        std::cout << f_vocab_->getWord(f) << " " << prob << " ";
        sum += prob;
      }
    }
    std::cout << "\t[sum: " << sum << "]" << endl;
  }
}

void LexiconSparse::write(std::string fn) const {
  CHECK_NOTNULL(e_vocab_);
  CHECK_NOTNULL(f_vocab_);

  misc::OFileStream ostr(fn);
  ostr.get() << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  for (size_t e = 0; e < e_vocab_->size(); ++e) {
    for (auto it = ef_counts_[e].begin(); it != ef_counts_[e].end(); ++it) {
      word_id_t f = it->first;
      score_t count = it->second;
      if (count > 0) {
        ostr.get() << e_vocab_->getWord(e) << " " << f_vocab_->getWord(f) << " "
                   << count << endl;
      }
    }
  }
}

void LexiconSparse::read(std::string fn, bool add_words) {
  CHECK_NOTNULL(e_vocab_);
  CHECK_NOTNULL(f_vocab_);
  ////////////////////////////////////////////////////////////////////////
  if (add_words) {
    misc::IFileStream istr(fn);
    for (std::string current_line; std::getline(istr.get(), current_line);) {
      std::vector<std::string> fields;
      boost::split(fields, current_line, boost::is_any_of("\t "));
      CHECK(fields.size() == 5 || fields.size() == 3);
      std::string e_str = fields[0];
      std::string f_str = fields[1];
      e_vocab_->addWord(e_str);
      f_vocab_->addWord(f_str);
    }
    e_vocab_->cacheTypes();
    f_vocab_->cacheTypes();
  }

  ////////////////////////////////////////////////////////////////////////
  misc::IFileStream istr(fn);

  ef_counts_.resize(e_vocab_->size());
  // ALSO FILLS lex_beam
  lex_beam_.clear();
  lex_beam_.resize(f_vocab_->size());

  size_t line_count = 0;
  size_t entries = 0;
  LOG(INFO) << "reading lexicon";

  CHECK_GT(f_vocab_->size(), 0);
  CHECK_GT(e_vocab_->size(), 0);

  LOG(INFO) << "eVocab.size = " << e_vocab_->size()
            << " / fVocab.size = " << f_vocab_->size();
  LOG(INFO) << "ef_counts.size = " << ef_counts_.size();
  LOG(INFO) << "ef_counts[0].size = " << ef_counts_[0].size();
  LOG(INFO) << "assuming format: 'E F SCORE'";

  size_t skipped_entries = 0;
  size_t e_unknown = 0;
  size_t f_unknown = 0;

  for (std::string current_line; std::getline(istr.get(), current_line);) {
    VLOG_EVERY_N(2, 50000) << line_count << " entries read" << std::endl;
    ++line_count;
    std::vector<std::string> fields;
    boost::split(fields, current_line, boost::is_any_of("\t "));

    CHECK(fields.size() == 5 || fields.size() == 3);

    word_id_t e, f;
    auto e_str = fields[0];
    auto f_str = fields[1];

    bool skip = false;
    if (!e_vocab_->containsWord(e_str)) {
      VLOG(1) << "lexicon contains word '" << e_str
              << "' that is not in eVocab. skipping...";
      ++e_unknown;
      skip = true;
    }

    if (!f_vocab_->containsWord(f_str)) {
      VLOG(1) << "lexicon contains word '" << f_str
              << "' that is not in fVocab. skipping...";
      ++f_unknown;
      skip = true;
    }

    if (skip) {
      ++skipped_entries;
      continue;
    }

    e = e_vocab_->getIdFail(e_str);
    f = f_vocab_->getIdFail(f_str);
    // TODO: THIS IS A HACK
    if (f >= lex_beam_.size()) lex_beam_.resize(f + 1);
    lex_beam_[f].push_back(e);

    ef_counts_[e][f] += std::stod(fields[2]);
    VLOG(3) << "adding " << e_str << " " << f_str << " " << fields[2];

    if (fields.size() == 5) {
      CHECK_GE(std::stoi(fields[3]), 0);
      CHECK_GE(std::stoi(fields[4]), 0);
    } else {
      //      e_counts[e] = 1;
      //      e_num[e] = 1;
    }
    entries++;
  }
  if (skipped_entries > 0) {
    LOG(WARNING) << "skipped " << skipped_entries << " entries: " << e_unknown
                 << " not known in eVocab, " << f_unknown
                 << " not known in fVocab";
  }

  LOG(INFO) << "reading lexicon done";
  LOG(INFO) << "eVocab.size = " << e_vocab_->size()
            << " / fVocab.size = " << f_vocab_->size();

  LOG(INFO) << "read " << entries << " entries" << endl;
}

const vector<std::unordered_map<word_id_t, score_t> >&
LexiconSparse::getCounts() const {
  return ef_counts_;
}

// max_evocab_id is needed, because the vocabulary might contain tokens that
// are only available in the reference output.
//
// the vocabulary has to be setup in a way that all reference-only tokens
// are stored with id's bigger than emax_vocab_id, otherwise we are implictly
// cheating
void LexiconSparse::addCountsFromLex(const LexiconSparse& lex,
                                     size_t max_evocab_id) {
  CHECK_EQ(e_vocab_, lex.e_vocab_);
  CHECK_EQ(f_vocab_, lex.f_vocab_);
  CHECK_NOTNULL(e_vocab_);
  CHECK_LE(max_evocab_id, lex.e_vocab_->size());
  for (word_id_t e = 0; e < max_evocab_id; ++e) {
    for (auto it = lex.ef_counts_[e].begin(); it != lex.ef_counts_[e].end();
         ++it) {
      word_id_t f = it->first;
      ef_counts_[e][f] += it->second;
    }
  }
}

void LexiconSparse::setDeltaLambda(score_t delta_lambda) {
  delta_lambda_ = delta_lambda;
}

void LexiconSparse::analyzeCorrectProb(const LexiconSparse& ref_lex,
                                       const string& eval_fn) const {
  std::vector<double> prob_thresholds = {0.001, 0.01, 0.03, 0.05,
                                         0.1,   0.2,  0.5};
  std::vector<double> correct_thresholds, total_thresholds;
  correct_thresholds.resize(prob_thresholds.size());
  total_thresholds.resize(prob_thresholds.size());
  double total_count = 0.0;
  double correct_prob = 0.0;
  double correct_entries = 0.0;
  size_t eval_entries = 0.0;
  // evaluate total fraction of counts (aka joint probability mass)
  for (word_id_t e : e_vocab_->getNormalWords()) {
    LOG_EVERY_N(INFO, 1000) << "working on e=" << e << " ("
                            << e_vocab_->getWord(e) << ")";
    total_count += 1.0;
    for (word_id_t f : f_vocab_->getNormalWords()) {
      double cur_prob = ::exp(score(f, e));
      bool ref_contained =
          ref_lex.getCounts()[e].find(f) != ref_lex.getCounts()[e].end();

      for (size_t i = 0; i < correct_thresholds.size(); ++i) {
        if (cur_prob > prob_thresholds[i]) {
          if (ref_contained) {
            ++correct_thresholds[i];
          }
          ++total_thresholds[i];
        }
      }
      if (ref_contained) {
        ++eval_entries;

        // LOG(INFO) << fVocab.getWord(f) << " " << eVocab.getWord(e);
        correct_prob += cur_prob;
      }
    }
  }

  misc::OFileStream ostr(eval_fn);
  ostr.get() << correct_prob / total_count << " " << correct_entries << " "
             << eval_entries;

  LOG(INFO) << "fraction of joint mass in correct mappngs = "
            << correct_prob / total_count;
  for (size_t i = 0; i < prob_thresholds.size(); ++i) {
    LOG(INFO) << correct_thresholds[i] << "/" << total_thresholds[i]
              << " entries with prob > " << prob_thresholds[i] << " correct";
    ostr.get() << " " << prob_thresholds[i] << " " << correct_thresholds[i]
               << " " << total_thresholds[i];
  }
  ostr.get() << std::endl;

  LOG(INFO) << "based on " << eval_entries << " evaluation entries";
}

void LexiconSparse::init_id() {
  CHECK_NOTNULL(e_vocab_);
  CHECK_NOTNULL(f_vocab_);

  // add all target words to source vocab
  for (word_id_t f = 1; f < f_vocab_->size(); ++f) {
    if (!e_vocab_->containsWord(f_vocab_->getWord(f))) {
      LOG(FATAL) << "Fatal: e_vocab does not contain '" << f_vocab_->getWord(f)
                 << "'" << endl;
    }
  }

  // add all target words to source vocab
  for (word_id_t e = 1; e < e_vocab_->size(); ++e) {
    if (!f_vocab_->containsWord(e_vocab_->getWord(e))) {
      f_vocab_->addWord(e_vocab_->getWord(e));
      LOG(WARNING) << "Warning: f_vocab does not contain '"
                   << e_vocab_->getWord(e) << "'. Added!" << endl;
    }
  }
  e_vocab_->cacheTypes();
  f_vocab_->cacheTypes();

  init();

  for (word_id_t f : f_vocab_->getNormalWords()) {
    for (word_id_t e : e_vocab_->getNormalWords()) {
      if (e_vocab_->getWord(e) == f_vocab_->getWord(f)) {
        add(e, f, 1.0);
      }
    }
  }
  learn();
  LOG(INFO) << "learned id lexicon";
}

void LexiconSparse::init_supervised(
    const std::vector<std::vector<word_id_t> >& snts,
    const std::vector<std::vector<word_id_t> >& snts_ref) {
  init();
  size_t n_snts = snts.size();
  CHECK_EQ(n_snts, snts_ref.size());
  for (size_t n = 0; n < n_snts; ++n) {
    size_t len = snts[n].size();
    CHECK_EQ(len, snts_ref[n].size());
    for (size_t i = 0; i < len; ++i) {
      add(snts_ref[n][i], snts[n][i], 1.0);
    }
  }
  learn();
  LOG(INFO) << "learned supervised lexicon";
}

void LexiconSparse::init_binomial(
    const KenLmWithVocab& lm,
    const std::vector<std::vector<word_id_t> >& snts) {

  init();

  //////////////////////////////////////////////////////////////////////////
  // generate source counts
  std::vector<size_t> counts(f_vocab_->size(), 0);
  size_t N = 0;
  for (const std::vector<word_id_t>& snt : snts) {
    for (word_id_t f : snt) {
      ++counts[f];
      ++N;
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // debug output
  VLOG(1) << "p(f) ////////////////////////////////";
  for (word_id_t f : f_vocab_->getNormalWords()) {
    VLOG(1) << f_vocab_->getWord(f) << " "
            << counts[f] / static_cast<score_t>(N);
  }
  KenLmWithVocab::state_t dummy_state;
  VLOG(1) << "p(e) ////////////////////////////////";
  for (word_id_t e : e_vocab_->getNormalWords()) {
    score_t score =
        LN_10 * lm.getScore(lm.getModel().NullContextState(), e, &dummy_state);
    VLOG(1) << e_vocab_->getWord(e) << " " << exp(score);
  }

  //////////////////////////////////////////////////////////////////////////
  // p(e|f) [f][e]
  std::vector<std::vector<score_t> > p_fe(
      f_vocab_->size(),
      std::vector<score_t>(e_vocab_->size(),
                           -std::numeric_limits<score_t>::infinity()));
  for (word_id_t f : f_vocab_->getNormalWords()) {
    score_t max = -std::numeric_limits<score_t>::infinity();
    for (word_id_t e : e_vocab_->getNormalWords()) {
      score_t lm_score = LN_10 * lm.getScore(lm.getModel().NullContextState(),
                                             e, &dummy_state);
      score_t counter_lm_score = log(1 - exp(lm_score));
      p_fe[f][e] = counts[f] * lm_score + (N - counts[f]) * counter_lm_score;
      if (p_fe[f][e] > max) {
        max = p_fe[f][e];
      }
    }
    score_t sum = 0.0;
    for (word_id_t e : e_vocab_->getNormalWords()) {
      p_fe[f][e] = exp(p_fe[f][e] - max);  // switching from log to lin domain
      sum += p_fe[f][e];
    }
    for (word_id_t e : e_vocab_->getNormalWords()) {
      p_fe[f][e] = p_fe[f][e] / sum;
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // p(f|e)
  for (word_id_t e : e_vocab_->getNormalWords()) {
    for (word_id_t f : f_vocab_->getNormalWords()) {
      add(e, f, (p_fe[f][e] * counts[f]) / N);
    }
  }
  learn();
}
