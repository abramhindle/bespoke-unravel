#define LN_10 2.302585093
#include <glog/logging.h>

DEFINE_uint64(jobs, 1, "divide corpus into N parts");
DEFINE_uint64(job, 1, "divide corpus into N parts, this is part M, with 1<=M<=N");

DEFINE_string(lm, "", "language model to load");
DEFINE_string(output_prefix, "output_prefix_", "prefix of output files");
DEFINE_double(lex_lambda, 0.9, "smoothing parameter for lexicon");
DEFINE_uint64(permutation_window, 0, "maximum allowed jumpsize. 0=monotone");
DEFINE_double(jump_penalty, 1.0, "penalty for jumps");

DEFINE_bool(insertions, false, "allow insertions");
DEFINE_double(insertion_penalty, 0.0, "penalty for insertions");
DEFINE_uint64(max_deletions, 0, "allow deletion");
DEFINE_double(deletion_penalty, 0.0, "penalty for deletions");

DEFINE_double(arc_max_score, 10000.0, "maximum arc score that is ok");

DEFINE_uint64(reserve_hyps, 10000, "reserve hyps in vector");
DEFINE_uint64(reserve_states, 10000, "reserve states in vector");

DEFINE_uint64(beam_size, 100, "beam size");
DEFINE_uint64(lex_beam_prepare, 5, "lexicon beam prepare");
DEFINE_uint64(lex_beam_size, 5, "lexicon beam size");
DEFINE_double(lm_weight, LN_10, "weight for language model (compensates log10 scores for lm vs ln (e) scores for lex)");
DEFINE_uint64(lm_beam_size, 50, "lm beam size");
DEFINE_uint64(insertion_beam_size, 50, "lm beam size");
DEFINE_uint64(max_insertions, 2, "maximum number of insertions");

DEFINE_uint64(every, 25, "debug output every X lines");
DEFINE_string(e, "", "e corpus to load");
DEFINE_string(f, "", "f corpus to load");
DEFINE_uint64(iter, 0, "current iteration");
DEFINE_uint64(iters, 30, "number of iterations");
DEFINE_uint64(num_threads, 1, "number of threads");
DEFINE_uint64(max_sentences, 1000000, "number of sentences to read");
DEFINE_double(sentence_factor, 1.2,
              "increase number of training "
              "sentences by this factor");
DEFINE_uint64(test_sentences, 100, "number of sentences to take for test set");
DEFINE_bool(write_fst, false, "write fsts?");

// needed if FST components are used
// still a little weird
DEFINE_bool(fst_error_fatal, false, "fst fix");
DEFINE_bool(fst_verify_properties, false, "fst fix");
DEFINE_bool(fst_align, false, "fst fix");
DEFINE_bool(fst_default_cache_gc, true, "fst fix");
DEFINE_int64(fst_default_cache_gc_limit, 1 << 20LL, "fst fix");

#include <fst/fstlib.h>
#include <fst/vector-fst.h>
#include <fst/shortest-path.h>
#include <fst/symbol-table.h>
#include <fst/fst.h>

#include <boost/algorithm/string.hpp>
#include <boost/chrono.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include <atomic>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <math.h>

// scoring
#include "cdec/utils/stringlib.h"
#include "cdec/utils/filelib.h"
#include "cdec/utils/tdict.h"
#include "cdec/mteval/ns_cer.h"
#include "cdec/mteval/ns_comb.h"
#include "cdec/mteval/ns_docscorer.h"
#include "cdec/mteval/ns_ext.h"
#include "cdec/mteval/ns.h"
#include "cdec/mteval/ns_ssk.h"
#include "cdec/mteval/ns_ter.h"

#include "misc.hh"
#include "vocab.hh"
#include "lexicon.hh"
#include "lexicon_sparse.hh"
#include "lm_beam.hh"
#include "mapping.hh"
#include "kenlm_with_vocab.hh"
#include "fst_permute.hh"
#include "fst_beamsearch.hh"
#include "chunk_beam.hh"

#include "em_unravel_helpers.hh"

using std::cerr;
using std::cout;
using std::endl;

// not sure if we should really run to maxSentenceLength_ or one less
bool UnfinishedHyps(const std::vector<std::vector<HypNode>> &hyps) {
  for (size_t i = 0; i < maxSentenceLength_; ++i) {
    if (hyps[i].size() > 0) return true;
  }
  return false;
}

std::string StacksToString(std::string name, std::vector<std::vector<HypNode>> &hyps,
                size_t sentence_len) {
  std::stringstream ss;
  ss << "stack " << name << ": ";
  for (size_t i = 0; i < sentence_len; ++i) ss << hyps[i].size() << " ";
  return ss.str();
}

inline void AddOrRecombine(fst::VectorFst<fst::LogArc> *lattice, HypNode &cur_hyp, HypToStateMap *hyp_to_state, std::vector<std::vector<HypNode>> *new_stack_ptr) {
  CHECK_NOTNULL(hyp_to_state);
  CHECK_NOTNULL(new_stack_ptr);
  auto &new_stack = *new_stack_ptr;

  typename HypToStateMap::const_iterator it = hyp_to_state->find(cur_hyp);
  if (it != hyp_to_state->end()) {
    size_t hyp_index = it->second;
    // recombination
    CHECK_LT(cur_hyp.cardinality, new_stack.size());
    CHECK_LT(hyp_index, new_stack[cur_hyp.cardinality].size());
    HypNode &recomb_hyp = new_stack[cur_hyp.cardinality][hyp_index];
    cur_hyp.output_fst_state = recomb_hyp.output_fst_state;
    CHECK_LT(cur_hyp.output_fst_state, lattice->NumStates());
    recomb_hyp.score = misc::add_log_scores(recomb_hyp.score, cur_hyp.score);
    CHECK(std::isfinite(recomb_hyp.score));
    CHECK(CanRecombine(cur_hyp, recomb_hyp));
  } else {
    size_t cur_id = new_stack[cur_hyp.cardinality].size();
    hyp_to_state->insert(typename HypToStateMap::value_type(cur_hyp, cur_id));
    cur_hyp.output_fst_state = lattice->AddState();
    new_stack[cur_hyp.cardinality].push_back(cur_hyp);
  }
  CHECK_LT(cur_hyp.pred_output_fst_state, lattice->NumStates());
  CHECK_LT(cur_hyp.output_fst_state, lattice->NumStates());
  lattice->AddArc(cur_hyp.pred_output_fst_state,
      fst::LogArc(cur_hyp.f, cur_hyp.e, cur_hyp.delta,
        cur_hyp.output_fst_state));
}

inline bool CheckHyp(fst::VectorFst<fst::LogArc> *lattice, const HypNode &cur_hyp) {

  // worst score we allow
  score_t max_allowed_score = (1+cur_hyp.cardinality) * 98;

  if (cur_hyp.score > max_allowed_score) {
    LOG(WARNING) << "SKIPPING HYP BECAUSE SCORE IS TOO BAD!";
    return false;
  }

  CHECK(std::isfinite(cur_hyp.score));

  CHECK_LT(cur_hyp.pred_output_fst_state, lattice->NumStates());
  if (FLAGS_max_deletions == 0) {
    CHECK_EQ(eVocab_.isNormal(cur_hyp.e), fVocab_.isNormal(cur_hyp.f));
    //CHECK(VocabHelper::CheckSpecialSymbols(cur_hyp.e, cur_hyp.f));
  }

  CHECK(!((cur_hyp.e == eVocab_.null()) &&
        (cur_hyp.f == fVocab_.null())));

  return true;
}

void ExpandNode(const CoverageVector &full_coverage, fst::VectorFst<fst::LogArc> *lattice, const HypNode &predecessor_hyp, word_id_t f, size_t pos, size_t jump, const std::set<word_id_t> &to_expand_set, std::vector<std::vector<HypNode>> *active_new_ptr, HypToStateMap *hyp_to_state, const KenLmWithVocab &lm) {
  CHECK_NOTNULL(active_new_ptr);
  auto &active_new = *active_new_ptr;

  // set up new hyp
  HypNode new_hyp = predecessor_hyp;
  new_hyp.f = f;
  new_hyp.pred_output_fst_state = predecessor_hyp.output_fst_state;

  // jump cost is constant for all expansion
  score_t score_jump = jump * FLAGS_jump_penalty;

  // check if this is an insertion
  new_hyp.coverage = predecessor_hyp.coverage;
  if (f != fVocab_.null()) {
    new_hyp.coverage.set(pos);
  } else {
    if (predecessor_hyp.num_insertions >= FLAGS_max_insertions) return;
    ++new_hyp.num_insertions;
  }

  // loop over all possible e extensions
  for (word_id_t e : to_expand_set) {

    new_hyp.e = e;

    if (e == eVocab_.null()) {
      new_hyp.cardinality = predecessor_hyp.cardinality;
      new_hyp.num_deletions = predecessor_hyp.num_deletions + 1;
    } else {
      new_hyp.cardinality = predecessor_hyp.cardinality + 1;
      new_hyp.num_deletions = predecessor_hyp.num_deletions;
    }

    if (new_hyp.num_deletions > FLAGS_max_deletions) continue;

    score_t score_lex = 0.0;
    CHECK_EQ(eVocab_.isNormalOrNull(e), fVocab_.isNormalOrNull(f)) << "e='" <<
      eVocab_.getWord(e) << "' f='" << fVocab_.getWord(f) <<"'";

    if (fVocab_.isNormalOrNull(f)) {
      score_lex = -lex_->score(f, e);
    } else {
      // for special symbols, 0.0 is fine
      score_lex = 0.0;
    }

    // lex score

    CHECK_LT(new_hyp.cardinality, active_new.size());

    // lm score
    score_t score_lm = 0.0;
    score_t score_del = 0.0;
    if (e == eVocab_.null()) {
      new_hyp.lm_state = predecessor_hyp.lm_state;
      score_lm = 0.0;
      score_del = FLAGS_deletion_penalty;
    } else {
      CHECK(lm.validWordId(e)) << "invalid word" << eVocab_.getWord(e) << " (" << e << ")";
      score_lm = - FLAGS_lm_weight * lm.getScore(predecessor_hyp.lm_state, e, &new_hyp.lm_state);
      score_del = 0.0;
    }

    // insertion score
    score_t score_ins = 0.0;
    if (f == fVocab_.null()) {
      score_ins = FLAGS_insertion_penalty;
      new_hyp.last_insertion = true;
    } else {
      score_ins = 0.0;
      new_hyp.last_insertion = false;
    }

    CHECK_GE(score_lm, 0.0);
    CHECK_GE(score_lex, 0.0);
    CHECK_GE(score_jump, 0.0);

    // total score
    new_hyp.delta = score_lm + score_lex + score_jump + score_ins + score_del;
    new_hyp.score = predecessor_hyp.score + new_hyp.delta;

    CHECK(std::isfinite(score_lm));
    CHECK(std::isfinite(score_lex));
    CHECK(std::isfinite(score_jump));
    CHECK(std::isfinite(score_ins));
    CHECK(std::isfinite(score_del));
    CHECK(std::isfinite(new_hyp.delta));
    CHECK(std::isfinite(new_hyp.score));

    CHECK_LT(new_hyp.delta, FLAGS_arc_max_score) << "arc_score for f='" << fVocab_.getWord(f)
      << "' e='" << eVocab_.getWord(e)
      << "' lm-state='" << lm.stateToString(predecessor_hyp.lm_state)
      << "' too high: delta="
      << new_hyp.delta << "= " << score_lm << "(lm) + "
      << score_lex << "(lex) + " << score_jump << "(jump)";

    // check if we want to add this hyp
    if (CheckHyp(lattice, new_hyp)) {
      AddOrRecombine(lattice, new_hyp, hyp_to_state, &active_new);
      if (new_hyp.coverage == full_coverage) {
        lattice->SetFinal(new_hyp.output_fst_state, 0.0);
      }
    }
  }
}

void PruneHyps(std::vector<std::vector<HypNode>> *active_new_ptr) {
  auto &active_new = *active_new_ptr;
  // sort all cards
  for (size_t c = 0; c < maxSentenceLength_; ++c) {
    size_t elements_to_keep =
      std::min(FLAGS_beam_size, static_cast<google::uint64>(active_new[c].size()));
    std::nth_element(active_new[c].begin(), active_new[c].begin() + elements_to_keep, active_new[c].end(), HypNodeCompareScore);
    active_new[c].resize(elements_to_keep);
  }
}

void BeamSearch(std::vector<std::vector<HypNode>> *active_ptr,
                std::vector<std::vector<HypNode>> *active_new_ptr,
                const std::vector<word_id_t> &input_sentence,
                KenLmWithVocab &lm, fst::VectorFst<fst::LogArc> *lattice) {
  CHECK_NOTNULL(active_ptr);
  CHECK_NOTNULL(active_new_ptr);
  CHECK_NOTNULL(lattice);
  CHECK_LT(input_sentence.size() + 1, maxSentenceLength_);

  lattice->DeleteStates();

  CoverageVector full_coverage = MakeFullCoverageVector(input_sentence.size());

  std::vector<std::vector<HypNode>> &active = *active_ptr;
  std::vector<std::vector<HypNode>> &active_new = *active_new_ptr;

  // init stacks
  /////////////////////////////////////////////////////////////////////////////
  for (size_t i = 0; i < maxSentenceLength_; ++i) {
    active[i].resize(0);
    active_new[i].resize(0);
  }

  // init active
  active[0].resize(1);
  active[0][0] = HypNode(lm);
  active[0][0].output_fst_state = lattice->AddState();
  lattice->SetStart(active[0][0].output_fst_state);
  CHECK_EQ(active[0][0].num_insertions, 0);

  /////////////////////////////////////////////////////////////////////////////

  HypToStateMap hyp_to_state;

  // unfinished hypotheses?
  while (UnfinishedHyps(active)) {

    // loop over all stacks
    for (size_t c = 0; c < maxSentenceLength_; ++c) {
      if (active[c].size() == 0) continue;
      // loop over all partial hyps
      for (size_t i = 0; i < active[c].size(); ++i) {
        const HypNode &cur_hyp = active[c][i];

        // add insertions
        if (!cur_hyp.last_insertion && FLAGS_insertions) {
          std::set<word_id_t> to_expand_set;
          lm.addCandidates(cur_hyp.lm_state, &to_expand_set, FLAGS_insertion_beam_size);
          size_t jump_size = 0;
          ExpandNode(full_coverage, lattice, cur_hyp, fVocab_.null(), 0, jump_size, to_expand_set, &active_new, &hyp_to_state, lm);
        }

        // loop over possible source words
        for (word_id_t pos = 0; pos < input_sentence.size(); ++pos) {
          if (!JumpPossible(cur_hyp, pos, input_sentence.size(), FLAGS_permutation_window)) {
            continue;
          }

          // NOTE: lex is not log(prob)
          size_t jump_size = JumpSize(cur_hyp, pos, input_sentence.size());
          word_id_t f = input_sentence[pos];

          // get candidates
          std::set<word_id_t> to_expand_set;

          if (!fVocab_.isNormal(f)) {
            AddSpecialCandidates(f, &to_expand_set, FLAGS_max_deletions);
          } else {
            lex_->addCandidates(f, &to_expand_set, FLAGS_lex_beam_size);
            lm.addCandidates(cur_hyp.lm_state, &to_expand_set, FLAGS_lm_beam_size);
          }

          ExpandNode(full_coverage, lattice, cur_hyp, f, pos, jump_size, to_expand_set, &active_new, &hyp_to_state, lm);

        }  // end loop source positions
      }    // end loop cards
    }  // end loop active

    PruneHyps(&active_new);

    // swap
    auto *old_active = &active;
    active = active_new;
    active_new = *old_active;

    // clear active_new
    for (size_t c = 0; c < maxSentenceLength_; ++c) {
      active_new[c].resize(0);
    }
  }  // end while

  for (size_t c = 0; c < maxSentenceLength_; ++c) {
    CHECK_EQ(active[c].size(), 0);
  }
}

std::string WriteViterbi(size_t num, const fst::VectorFst<fst::StdArc> &input,
                  const bool is_test, statistics *s) {
  CHECK_NOTNULL(s);
  const std::vector<word_id_t> &ref = e_sentences_idx_[num];

  std::stringstream ss;

  int64 state = input.Start();
  fst::TropicalWeight score = 0.0;
  CHECK_GE(state, 0);
  ss << num << "\t";

  // int needed for evaluator
  std::vector<int> f, e;

  while (input.Final(state) == fst::StdArc::Weight::Zero()) {
    for (fst::ArcIterator<fst::VectorFst<fst::StdArc>> aiter(input, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      score = fst::Times(score, arc.weight);

      f.push_back(arc.ilabel);
      e.push_back(arc.olabel);

      ss << fVocab_.getWord(arc.ilabel) << "|" << eVocab_.getWord(arc.olabel)
         << " ";

      // different counters depending on test or train part of corpus
      if (fVocab_.getWord(arc.ilabel) != "UNK") {
        if (fVocab_.getWord(arc.ilabel) == eVocab_.getWord(arc.olabel)) {
          if (is_test)
            s->test_tokens_correct++;
          else
            s->train_tokens_correct++;
        }
        if (is_test)
          s->test_tokens_total++;
        else
          s->train_tokens_total++;
      }
      state = arc.nextstate;
    }
  }

  SufficientStats stats;
  // LOG(INFO) << "num=" << num << " sent_size " << e_sentences_idx_.size() <<
  // endl;

  size_t correct = 0;
  size_t total = 0;
  for (size_t i = 0; i < std::min(e.size(), ref.size()); ++i) {
    if (static_cast<int>(ref[i]) == e[i]) ++correct;
    ++total;
  }

  std::vector<int> reference(e_sentences_idx_[num].begin(),
                             e_sentences_idx_[num].end());
  std::vector<std::vector<int>> refs;
  refs.push_back(reference);
  EvaluationMetric *bleuMetric = EvaluationMetric::Instance("IBM_BLEU");
  boost::shared_ptr<SegmentEvaluator> segmentEvaluator =
      bleuMetric->CreateSegmentEvaluator(refs);
  segmentEvaluator->Evaluate(e, &stats);

  double bleu = 100.0 * bleuMetric->ComputeScore(stats);
  ss << score << " " << bleu << " " << is_test;

  s->bleu_total += bleu;

  return ss.str();
}

void null_stats(fst::VectorFst<fst::StdArc> *input, statistics *stats) {
  CHECK_NOTNULL(input);

  const int numStates = input->NumStates();

  for (int64 state = 0; state < numStates; state++) {
    for (fst::MutableArcIterator<fst::VectorFst<fst::StdArc>> aiter(input,
                                                                    state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      CHECK_LT(state, numStates);
      CHECK_LT(arc.nextstate, numStates);
      CHECK_LT(input->Start(), numStates);
      CHECK_LT(arc.olabel, static_cast<int>(eVocab_.size()));
      CHECK_LT(arc.ilabel, static_cast<int>(fVocab_.size()));
      stats->deletions_total += (arc.olabel == static_cast<int>(eVocab_.null()));
      stats->insertions_total += (arc.ilabel == static_cast<int>(fVocab_.null()));
    }
  }
}

// use lattice to accumulate posteriors into lexicon
// use thread_id to pass to lexicon
void Accumulate(const size_t thread_id, fst::VectorFst<fst::LogArc> *input) {
  const int numStates = input->NumStates();
  std::vector<fst::LogArc::Weight> alpha(numStates, 0);
  std::vector<fst::LogArc::Weight> beta(numStates, 0);
  fst::ShortestDistance<fst::LogArc>(*input, &alpha, false);
  fst::ShortestDistance<fst::LogArc>(*input, &beta, true);

  std::vector<score_t> total_f;
  total_f.resize(fVocab_.size(), 0.0);

  for (int64 state = 0; state < numStates; state++) {
    for (fst::MutableArcIterator<fst::VectorFst<fst::LogArc>> aiter(input,
                                                                    state);
         !aiter.Done(); aiter.Next()) {
      const fst::LogArc &arc = aiter.Value();
      CHECK_LT(state, numStates);
      CHECK_LT(arc.nextstate, numStates);
      CHECK_LT(input->Start(), numStates);
      CHECK_LT(arc.olabel, static_cast<int>(eVocab_.size()));
      CHECK_LT(arc.ilabel, static_cast<int>(fVocab_.size()));

      word_id_t e = arc.olabel;
      if (!eVocab_.isNormalOrNull(e)) continue;

      word_id_t f = arc.ilabel;
      if (!fVocab_.isNormalOrNull(f)) continue;

      double posterior =
          exp(-(alpha[state].Value() + arc.weight.Value() +
                beta[arc.nextstate].Value() - beta[input->Start()].Value()));
      if (!std::isfinite(posterior)) {
        LOG(FATAL) << "posterior for " << eVocab_.getWord(e) << " - " << fVocab_.getWord(f)
                   << " is not finite." << endl
                   << " alpha=" << alpha[state].Value()
                   << " arc=" << arc.weight.Value()
                   << " beta=" << beta[arc.nextstate].Value()
                   << " beta[begin]=" << beta[input->Start()].Value();
      }

      accumulation_lex_[thread_id]->add(e, f, posterior);
    }
  }
}

void InitializeStacks(std::vector<std::vector<HypNode>> *stack) {
  CHECK_NOTNULL(stack);
  stack->resize(maxSentenceLength_);
  for (size_t i = 0; i < stack->size(); ++i) {
    (*stack)[i].reserve(FLAGS_reserve_hyps);
  }
}

void UpdateStatusBar(size_t thread_id, size_t sent, const fst::VectorFst<fst::LogArc> &lattice, const fst::VectorFst<fst::StdArc> &shortest_path, statistics *s) {
  if (s->sentences_done % FLAGS_every == 0) {
    cerr << "\r";
    cerr << std::setprecision(2) << std::fixed;
    cerr << "[iter " << std::setw(2) << FLAGS_iter << "] [";
    cerr << "sent " << std::setw(5) << sent << " | ";
    cerr << std::setw(6) << s->sentences_done << "/" << FLAGS_test_sentences
      << "/" << FLAGS_max_sentences << "] [";
    cerr << std::setw(3) << thread_id << "/" << FLAGS_num_threads << "] [";
    cerr << "#states=" << std::setw(5) << lattice.NumStates() << ", ";
    cerr << "viterbi #states " << std::setw(5) << shortest_path.NumStates()
      << "] [";
    cerr << "ins " << std::setw(5)
      << static_cast<double>(s->insertions_total) /
      static_cast<double>(s->sentences_done) << "] [";
    cerr << "del " << std::setw(5)
      << static_cast<double>(s->deletions_total) /
      static_cast<double>(s->sentences_done) << "] [";
    cerr << "bleu_avg " << std::setw(5)
      << static_cast<double>(s->bleu_total) /
      static_cast<double>(s->sentences_done) << "%] [";
    cerr << "acc_test " << std::setw(5)
      << 100.0 * static_cast<double>(s->test_tokens_correct) /
      static_cast<double>(s->test_tokens_total) << "%] [";
    cerr << "acc_train " << std::setw(5)
      << 100.0 * static_cast<double>(s->train_tokens_correct) /
      static_cast<double>(s->train_tokens_total) << "%]";
    cerr << endl;
  }
}

void Work(size_t thread_id, misc::OFileStream *ostr, statistics *s) {

  // fetch sentence to work on
  std::size_t sent = std::atomic_fetch_add(&s->cur_sent, std::size_t(1));

  // prepare lattices
  fst::VectorFst<fst::LogArc> lattice;
  fst::VectorFst<fst::StdArc> lattice_std, shortest_path;
  lattice.ReserveStates(FLAGS_reserve_states);

  // prepare stacks
  std::vector<std::vector<HypNode>> active, active_new;
  InitializeStacks(&active);
  InitializeStacks(&active_new);


  while (sent < FLAGS_max_sentences) {

    const std::vector<word_id_t> &sentence_vec = f_sentences_idx_[sent];

    // skip specific sentences depending on job configuration
    if (sentence_vec.size() <= maxSentenceLength_ && ((sent % FLAGS_jobs + 1) == FLAGS_job)) {
      BeamSearch(&active, &active_new, sentence_vec, *klm_, &lattice);
      fst::Connect(&lattice);

      bool is_test_sentence = true;

      // do not learn with first test_sentences
      if (sent >= FLAGS_test_sentences) {
        Accumulate(thread_id, &lattice);
        is_test_sentence = false;
      }

      // get shortest path
      lattice_std.DeleteStates();
      shortest_path.DeleteStates();
      fst::Cast(lattice, &lattice_std);
      fst::ShortestPath(lattice_std, &shortest_path, 1);

      if (FLAGS_write_fst) {
        // use lattice_std so that we can apply shortest path on the command line
        lattice_std.SetInputSymbols(fVocab_.getFstSymbolTable());
        lattice_std.SetOutputSymbols(eVocab_.getFstSymbolTable());
        lattice_std.Write("lattice." + std::to_string((long long)(sent)) + ".fst");
      }

      std::string stat_string = WriteViterbi(sent, shortest_path, is_test_sentence, s);

      // critical section
      {
        std::unique_lock<std::mutex> lck(mtx_);
        null_stats(&shortest_path, s);
        UpdateStatusBar(thread_id, sent, lattice, shortest_path, s);
        ostr->get() << stat_string << endl;
        if (s->sentences_done % 10 == 0) ostr->get().flush();
      }
    } 

    // fetch next sentence
    sent = std::atomic_fetch_add(&s->cur_sent, std::size_t(1));
    s->sentences_done++;
  }
}

int main(int argc, char **argv) {
  INIT_MAIN("read common crawl format and index words\n");

  LOG(INFO) << "job=" << FLAGS_job << ", jobs=" << FLAGS_jobs;
  std::string job_str = misc::intToStrZeroFill(FLAGS_job, 3);
  std::string jobs_str = misc::intToStrZeroFill(FLAGS_jobs, 3);

  // get lm states
  LOG(INFO) << "setup lm and lm_beam";
  std::vector<std::vector<word_id_t>> states;

  klm_ = new KenLmWithVocab();
  klm_->read(FLAGS_lm.c_str(), &eVocab_);

  //lmbeam_ = new LmBeam(eVocab_, fVocab_, FLAGS_lm_beam_size);
  //lmbeam_->init(*klm_, states);

  LOG(INFO) << "read f corpus";
  LOG(INFO) << FLAGS_f;
  f_sentences_ = misc::readCorpus(FLAGS_f, FLAGS_max_sentences);
  LOG(INFO) << "read e corpus";
  LOG(INFO) << FLAGS_e;
  e_sentences_ = misc::readCorpus(FLAGS_e, FLAGS_max_sentences);
  LOG(INFO) << "read lm corpus";

  LOG(INFO) << "setup fvocab from " << f_sentences_.size() << " sentences"
            << endl;
  for (size_t i = 0; i < f_sentences_.size(); ++i) {
    f_sentences_idx_.push_back(fVocab_.getIdsAdd(f_sentences_[i] + " </s>"));
  }
  LOG(INFO) << "f_sentences_idx.size() = " << f_sentences_idx_.size();

  // this is not really efficient, but is fast enough for now
  LOG(INFO) << "eVocab has size " << eVocab_.size();
  LOG(INFO) << "fVocab has size " << fVocab_.size();

  // need to add
  eVocab_.addWord("<s>");
  eVocab_.addWord("</s>");
  eVocab_.addWord("<null>");
  // need to add
  fVocab_.addWord("<null>");
  fVocab_.addWord("<s>");
  fVocab_.addWord("</s>");

  eVocab_.cacheTypes();
  fVocab_.cacheTypes();

  klm_->prepare_known_states();
  klm_->prepare_successors(std::max(FLAGS_lm_beam_size, FLAGS_insertion_beam_size));

  LOG(INFO) << "setup lexicon";
  lex_ = new LexiconSparse(&eVocab_, &fVocab_, FLAGS_lex_beam_prepare, FLAGS_lex_lambda);
  for (size_t i=0;i<FLAGS_num_threads;++i) {
    accumulation_lex_.push_back(new LexiconSparse(&eVocab_, &fVocab_, FLAGS_lex_beam_prepare, FLAGS_lex_lambda));
  }

  LOG(INFO) << "looking for lexicons 'xxx_lex.gz' from previous runs";
  size_t iter_ok = 0;
  bool found_lex = false;
  for (size_t i = 0; i < FLAGS_iters; ++i) {
    std::string fn = FLAGS_output_prefix + misc::intToStrZeroFill(i, 3) + "_lex.gz";
    if (misc::getFileSize(fn) > 0) {
      iter_ok = i;
      found_lex = true;
    }
  }

  if (found_lex) {
    LOG(INFO) << "found lexicon at iteration = " << iter_ok;
    std::string iter_ok_str = misc::intToStrZeroFill(iter_ok, 3);
    lex_->read(FLAGS_output_prefix + iter_ok_str + "_lex.gz", false);
    lex_->learn();

    std::string iter_ok_out_str = iter_ok_str + "_" + job_str + "_of_" + jobs_str;
    lex_->write(FLAGS_output_prefix + iter_ok_out_str + "_lex.read.gz");
    FLAGS_iter = iter_ok + 1;
  } else {
    LOG(INFO) << "no lexicon found";
  }

  // ADD LAST, after we got all words
  for (size_t i = 0; i < e_sentences_.size(); ++i) {
    e_sentences_idx_.push_back(eVocab_.getIdsAdd(e_sentences_[i] + " </s>"));
  }
  LOG(INFO) << "e_sentences_idx.size() = " << e_sentences_idx_.size();

  // we invalidated the cache while possibly adding new reference tokens
  eVocab_.cacheTypes();

  statistics s;

  CHECK_LE(FLAGS_test_sentences, FLAGS_max_sentences) << "need more test sentences than sentences in total.";

  if (FLAGS_test_sentences >= 0.5 * FLAGS_max_sentences) {
    LOG(WARNING) << "more than 50% of the sentences are declared as "
                 "test-set. THIS WILL MOST CERTAINLY NOT BE RIGHT!";
  }

  // iterations
  while (FLAGS_iter < FLAGS_iters) {
    boost::chrono::time_point<boost::chrono::process_user_cpu_clock>
        iter_user_start = boost::chrono::process_user_cpu_clock::now();
    boost::chrono::time_point<boost::chrono::system_clock> iter_wall_start =
        boost::chrono::system_clock::now();

    LOG(INFO) << "iteration " << FLAGS_iter << " of " << FLAGS_iters;

    std::string iter_str = misc::intToStrZeroFill(FLAGS_iter, 3);

    if (FLAGS_jobs > 1) {
      iter_str = iter_str + "_" + job_str + "_of_" + jobs_str;
    }

    if (FLAGS_max_sentences > f_sentences_.size()) {
      LOG(INFO) << "reached full corpus length: setting max_sentences = " <<
        FLAGS_max_sentences << " to " << f_sentences_.size() << endl;
      FLAGS_max_sentences = f_sentences_.size();
    }

    misc::OFileStream viterbi_ostr(FLAGS_output_prefix + iter_str + "_" + "viterbi.gz");
    LOG(INFO) << "setup lex beam";
    lex_->setupBeam(klm_->getFirstInvalidVocabId());

    s.reset();

    LOG(INFO) << "start " << FLAGS_num_threads << " decoding threads";
    std::vector<std::thread> threads;
    for (size_t i = 0; i < FLAGS_num_threads; ++i)
      threads.push_back(std::thread(Work, i, &viterbi_ostr, &s));
    for (auto &t : threads) t.join();
    cerr << endl;
    LOG(INFO) << "joined " << FLAGS_num_threads << " threads";

    auto parallel_wall_end = boost::chrono::system_clock::now();
    boost::chrono::duration<double> beam_time =
        parallel_wall_end - iter_wall_start;

    auto learn_wall_start = boost::chrono::system_clock::now();
    lex_->init();

    // accumulate counts and reset lexicons from individual threads for next iteration
    for (size_t i=0;i<FLAGS_num_threads;++i) {
      lex_->addCountsFromLex(*(accumulation_lex_[i]), klm_->getFirstInvalidVocabId());
      accumulation_lex_[i]->init();
    }
    lex_->write(FLAGS_output_prefix + iter_str + "_counts.gz");
    lex_->learn();
    auto learn_wall_end = boost::chrono::system_clock::now();
    boost::chrono::duration<double> learn_time =
        learn_wall_end - learn_wall_start;

    lex_->write(FLAGS_output_prefix + iter_str + "_lex.gz");

    double mem_vm, mem_rss;
    misc::process_mem_usage(mem_vm, mem_rss);

    // log cpu time
    boost::chrono::time_point<boost::chrono::process_user_cpu_clock>
        iter_user_end = boost::chrono::process_user_cpu_clock::now();
    boost::chrono::time_point<boost::chrono::system_clock> iter_wall_end =
        boost::chrono::system_clock::now();
    auto iter_user_ms =
        boost::chrono::duration_cast<boost::chrono::milliseconds>(
            iter_user_end - iter_user_start);
    auto iter_wall_ms =
        boost::chrono::duration_cast<boost::chrono::milliseconds>(
            iter_wall_end - iter_wall_start);
    LOG(INFO) << "iteration used [" << iter_user_ms.count()
              << " ms] CPU time and [" << iter_wall_ms.count()
              << " ms] WALL TIME";
    LOG(INFO) << "parallelism = "
              << double(iter_user_ms.count()) / double(iter_wall_ms.count())
              << "x vs. threads=" << FLAGS_num_threads;

    // add beam_cpu_time to output
    misc::OFileStream log_ostr(FLAGS_output_prefix + iter_str + "_log.gz");
    log_ostr.get()
        << "beam_size\tlm_beam_size\tlex_beam_size\t";
    log_ostr.get() << "iter\ttrain_tokens_correct\ttrain_tokens_total\ttrain_"
                      "acc\ttest_tokens_correct\ttest_tokens_total\ttest_"
                      "acc\tbeam_wall_time\tlearn_time\tmem_vm\tmem_rss"
                   << endl;
    log_ostr.get() << FLAGS_beam_size << "\t" << FLAGS_lm_beam_size << "\t"
                   << FLAGS_lex_beam_size << "\t";
    log_ostr.get() << FLAGS_iter << "\t";
    log_ostr.get() << s.train_tokens_correct << "\t" << s.train_tokens_total
                   << "\t" << static_cast<double>(s.train_tokens_correct) /
                                  static_cast<double>(s.train_tokens_total)
                   << "\t";
    log_ostr.get() << s.test_tokens_correct << "\t" << s.test_tokens_total
                   << "\t" << static_cast<double>(s.test_tokens_correct) /
                                  static_cast<double>(s.test_tokens_total)
                   << "\t";
    log_ostr.get() << beam_time.count() << "\t" << learn_time.count() << "\t";
    log_ostr.get() << mem_vm / 1024 << "\t" << mem_rss / 1024 << endl;
    log_ostr.get().flush();
    LOG(INFO) << "test_tokens " << s.test_tokens_correct << "/"
              << s.test_tokens_total << " = "
              << static_cast<double>(s.test_tokens_correct) /
                     static_cast<double>(s.test_tokens_total)
              << " train_tokens " << s.train_tokens_correct << "/"
              << s.train_tokens_total << " = "
              << double(s.train_tokens_correct) / double(s.train_tokens_total)
              << " took " << beam_time.count() << "s"
              << " + " << learn_time.count() << "s"
              << " mem " << mem_vm / 1024 << "MB " << mem_rss / 1024 << "MB"
              << endl;
    FLAGS_iter++;
    FLAGS_max_sentences *= FLAGS_sentence_factor;
  }

  return 0;
}
