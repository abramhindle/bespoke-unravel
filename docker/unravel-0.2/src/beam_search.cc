#include "beam_search.hh"

using std::endl;

#define START_CARD 0

#define WRITE_BEST_HYPO(stream, width, card, thresh, histo, mapping, pred, \
                        n_hyps, diff_worst)                                \
  stream << std::setw(width) << std::left << card;                         \
  stream << std::setw(width) << std::left << thresh;                       \
  stream << std::setw(width) << std::left << histo;                        \
  stream << std::setw(width) << std::left << mapping;                      \
  stream << std::setw(width) << std::left << pred;                         \
  stream << std::setw(width) << std::left << n_hyps;                       \
  stream << std::setw(width) << std::left << diff_worst << endl;

BeamSearch::BeamSearch(const Vocab& f_vocab, const Vocab& e_vocab,
                       const std::vector<std::string>& special_words)
    : f_vocab_(f_vocab),
      e_vocab_(e_vocab),
      special_words_(special_words),
      option_searchlog_(option_searchlog::NO),
      current_card(START_CARD),
      current_hyps(1),
      running(true) {}

BeamSearch::~BeamSearch() {}

void BeamSearch::setSearchLog(
    option_searchlog::OPTION_SEARCHLOG option_searchlog,
    std::ostream* search_log_os) {
  search_log_os_ = search_log_os;
  option_searchlog_ = option_searchlog;
}

void BeamSearch::init_bodies() {
  LOG(INFO) << "initializing bodies" << endl;

  CHECK_EQ(sourceVector.size(), targetVector.size());

  for (size_t i = 0; i < sourceVector.size(); ++i) {
    for (size_t j = 0; j < MAX_LEN; ++j) {
      sourceVector[i].e_word_count[j] = 0;
      targetVector[i].e_word_count[j] = 0;
    }
  }

  for (size_t i = 0; i < hypVector.size(); ++i) {
    hypVector[i].parent_id = INVALID_PARENT;
  }
}

void BeamSearch::add_fixed_words() {
  for (size_t j = 0; j < MAX_LEN; ++j) sourceVector[0].e_word_count[j] = 0;

  LOG(INFO) << "adding fixed words" << endl;
  for (size_t i = 0; i < special_words_.size(); ++i) {
    std::string cur_word = special_words_[i];
    size_t e_word = e_vocab_.getId(cur_word);
    size_t f_word = f_vocab_.getId(cur_word);
    LOG(INFO) << "thinking about FIXING WORD F=" << f_word << " (" << cur_word
              << ")"
              << " TO E=" << e_word << " (" << cur_word << ") ...";

    if (special_f_words.count(f_word) == 0 &&
        special_e_words.count(e_word) == 0) {
      LOG(INFO) << "FIXING! for first " << sourceVector.size() << " hyps"
                << endl;
      // store for fast lookup
      special_f_words.insert(f_word);
      special_e_words.insert(e_word);

      // initialize bodies
      for (size_t j = 0; j < sourceVector.size(); ++j) {
        sourceVector[j].word_pairs[current_card].e = e_word;
        sourceVector[j].word_pairs[current_card].f = f_word;
        sourceVector[j].e_word_count[e_word]++;
        sourceVector[j].cor_count = current_card + 1;
      }
      ++current_card;
    } else {
      LOG(INFO) << "NOT FIXING! FIXING WOULD NOT BE UNIQUE" << endl;
    }
  }
}

void BeamSearch::fastForward(const ExtensionOrder& extOrder, std::istream* is,
                             std::vector<Feature*> features) {
  LOG(INFO) << "fast forwarding" << endl;

  // analyze pruning sizes
  LOG(INFO) << "resizing source vector" << endl;
  // need to resize on the fly
  sourceVector.resize(10000, partial_hyp_body(features.size()));
  // need to resize on the fly
  targetVector.resize(10000, partial_hyp_body(features.size()));
  LOG(INFO) << "done" << endl;

  size_t card_max = f_vocab_.size();

  // check if we compiled with enough maximum length
  CHECK_GT(MAX_LEN, card_max);

  init_bodies();

  CHECK_EQ(current_card,  START_CARD);
  CHECK_EQ(current_hyps, 1);

  add_fixed_words();
  size_t initialized_up_to = current_card;

  std::vector<std::vector<partial_hyp_head> > search_tree;

  std::string line;
  std::istringstream in;

  // read search log
  while (std::getline(*is, line)) {
    in.clear();
    in.str(line);
    size_t in_current_card, in_id, in_parent_id;
    std::string in_source_string, in_target_string;
    score_t in_score;
    in >> in_current_card >> in_id >> in_parent_id >> in_source_string >>
        in_target_string >> in_score;

    CHECK_GT(in_current_card, 0);
    CHECK_GE(in_current_card, current_card);
    CHECK_LE(in_score, 0);

    size_t in_source_num = f_vocab_.getId(in_source_string);
    size_t in_target_num = e_vocab_.getId(in_target_string);

    if (in_source_num == VOCAB_WORD_NOT_FOUND) {
      LOG(FATAL) << "UNKNOWN CIPHER WORD FOUND IN SEARCH LOG AT CARDINALITY "
                << in_current_card << " ID " << in_id << " WITH SCORE "
                << in_score << ": '" << in_source_string << "'" << endl;
    }

    if (in_target_num == VOCAB_WORD_NOT_FOUND) {
      LOG(FATAL) << "UNKNOWN PLAIN WORD FOUND IN SEARCH LOG AT CARDINALITY "
                << in_current_card << " ID " << in_id << " WITH SCORE "
                << in_score << ": '" << in_target_string << "'" << endl;
    }

    // resize tree if necessary
    // in_current_card is the cardinality, so the index to be addressed
    // is in_current_card-1
    if (in_current_card > search_tree.size()) {
      search_tree.resize(in_current_card);
    }
    if (in_id >= search_tree[in_current_card - 1].size()) {
      search_tree[in_current_card - 1].resize(in_id + 1, features.size());
    }

    search_tree[in_current_card - 1][in_id].e = in_target_num;
    search_tree[in_current_card - 1][in_id].f = in_source_num;
    search_tree[in_current_card - 1][in_id].score = in_score;
    search_tree[in_current_card - 1][in_id].parent_id = in_parent_id;
  }

  // we require a static size of sourceVector above...
  // at least make sure that it's big enough
  CHECK_GT(sourceVector.size(), search_tree.size());
  CHECK_GT(targetVector.size(), search_tree.size());

  // now initialize sourceVector
  if (search_tree.size() > 0) {
    current_hyps = search_tree[search_tree.size() - 1].size();
    current_card = search_tree.size() - 1;

    LOG(INFO) << "Done: Read " << search_tree.size() << " levels from search log"
          << endl;
    LOG(INFO) << "Found " << current_hyps << " final hypotheses of cardinality "
          << current_card << endl;

    for (size_t i = 0; i < current_hyps; i++) {
      size_t hyp_id = i;
      sourceVector[i].score = search_tree[search_tree.size() - 1][i].score;
      // start with j=search_tree.size()-1
      for (size_t j = search_tree.size(); j-- > initialized_up_to + 1;) {
        partial_hyp_head& cur_head = search_tree[j][hyp_id];

        LOG(INFO) << "initializing sourceVector[" << i << "].word_pairs[" << j
              << "].e = " << size_t(cur_head.e) << " ("
              << e_vocab_.getWord(cur_head.e) << ")" << endl;
        LOG(INFO) << "initializing sourceVector[" << i << "].word_pairs[" << j
              << "].f = " << size_t(cur_head.f) << " ("
              << f_vocab_.getWord(cur_head.f) << ")" << endl;
        LOG(INFO) << "initializing sourceVector[" << i << "].e_word_count["
              << size_t(cur_head.e)
              << "]=" << size_t(sourceVector[i].e_word_count[cur_head.e])
              << "+1" << endl;

        sourceVector[i].word_pairs[j].e = cur_head.e;
        sourceVector[i].word_pairs[j].f = cur_head.f;
        sourceVector[i].e_word_count[cur_head.e]++;
        hyp_id = cur_head.parent_id;
      }
    }

    // debug what is going on with sourceVector initializiation
    for (size_t i = 0; i < 3; ++i) {
      size_t e = sourceVector[0].word_pairs[i].e;
      size_t f = sourceVector[0].word_pairs[i].f;
      LOG(INFO) << "initializing sourceVector[" << 0 << "].word_pairs[" << i
            << "].e = " << size_t(e) << " (" << e_vocab_.getWord(e) << ")"
            << endl;
      LOG(INFO) << "initializing sourceVector[" << 0 << "].word_pairs[" << i
            << "].f = " << size_t(f) << " (" << f_vocab_.getWord(f) << ")"
            << endl;
      LOG(INFO) << "initializing sourceVector[" << 0 << "].e_word_count[" << e
            << "] = " << size_t(sourceVector[0].e_word_count[e]) << endl;
    }

    LOG(INFO) << "checking extension order" << endl;

    // setup set of covered words
    std::set<size_t> coveredWords;
    for (size_t i = 0; i < current_card + 1; ++i) {
      coveredWords.insert(sourceVector[0].word_pairs[i].f);
    }

    // check each entry of extension order (NO current_card+1 HERE??)
    for (word_id_t i = 0; i < current_card; ++i) {
      word_id_t next_f = extOrder.getWordAt(i);
      if (coveredWords.count(next_f) == 0) {
        LOG(INFO) << "ERROR: EXTENSION ORDER DOES NOT FIT INPUT SEARCH LOG:"
              << " INITIALIZED HYPS DO NOT COVER '" << f_vocab_.getWord(next_f)
              << "' (" << next_f << ")" << endl;

        for (size_t j = 0; j < current_card + 1; ++j) {
          LOG(INFO) << "COVERED: '"
                << f_vocab_.getWord(sourceVector[0].word_pairs[j].f) << "' ("
                << size_t(sourceVector[0].word_pairs[j].f) << ")" << endl;
        }
        assert(false);
      }
    }
  } else {
    LOG(INFO) << "read 0 levels from search log" << endl;
  }
}

std::vector<Mapping> BeamSearch::search(
    const ExtensionOrder& extOrder, const std::vector<size_t>& limit_e,
    std::vector<Constraint*> constraints, std::vector<Feature*> features,
    const ReferenceMapping* referenceMapping,
    const std::vector<score_t>& pruneThresh,
    const std::vector<size_t>& pruneMinHyp,
    const std::vector<size_t>& pruneHist,
    const std::vector<size_t>& prune_mapping,
    const std::vector<size_t>& prune_predecessor,
    const std::string& best_hyp_filename,
    bool printBest) {

  size_t card_max = f_vocab_.size();

  // reuse
  std::vector<partial_hyp_body> tmp;

  // analyze pruning sizes
  CHECK_GE(pruneHist.size(), card_max);
  size_t pruneHistMax = 0;
  for (size_t i = 0; i < pruneHist.size(); ++i) {
    if (pruneHist[i] > pruneHistMax) pruneHistMax = pruneHist[i];
  }

  LOG(INFO) << "size of full hypothesis = " << sizeof(partial_hyp_body) << " bytes"
        << endl;
  LOG(INFO) << "size of hypothesis head = " << sizeof(partial_hyp_head) << " bytes"
        << endl;

  double sourceVectorMB =
      double(2 * pruneHistMax * sizeof(partial_hyp_body)) / (1024.0 * 1024.0);
  double hypVectorMB =
      double(2 * pruneHistMax * e_vocab_.size() * sizeof(partial_hyp_head)) /
      (1024.0 * 1024.0);

  LOG(INFO) << "pruneHistMax=" << pruneHistMax << endl;
  LOG(INFO) << "e_vocab__size=" << e_vocab_.size() << endl;

  LOG(INFO) << "MAX_LEN=" << MAX_LEN << endl;
  LOG(INFO) << "card_max=" << card_max << endl;

  LOG(INFO) << "allocating sourceVector and targetVector with " << 2 * pruneHistMax
        << " elements =~ " << sourceVectorMB << "MB each" << endl;
  LOG(INFO) << "allocating hypVector with " << 2 * pruneHistMax * e_vocab_.size()
        << " elements =~" << hypVectorMB << "MB" << endl;

  LOG(INFO) << "now allocating sourceVector..." << endl;
  sourceVector.resize(pruneHistMax + 1, partial_hyp_body(features.size()));

  LOG(INFO) << "now allocating targetVector..." << endl;
  targetVector.resize(pruneHistMax + 1, partial_hyp_body(features.size()));

  LOG(INFO) << "now allocating hypVector..." << endl;
  hypVector.resize(pruneHistMax * e_vocab_.size() + 1,
                   partial_hyp_head(features.size()));

  LOG(INFO) << "starting..." << endl;

  CHECK_LE(card_max, MAX_LEN);
  CHECK_LE(f_vocab_.size(), MAX_LEN);
  CHECK_LE(e_vocab_.size(), MAX_LEN);

  // need to do that anyway
  for (size_t i = 0; i < hypVector.size(); ++i) {
    hypVector[i].parent_id = INVALID_PARENT;
  }

  // add fixed words
  // ToDo: fix dangerous comparison
  if (current_card == START_CARD) {
    init_bodies();
    add_fixed_words();
  } else {
    LOG(INFO) << "detected " << current_hyps
          << " previously initialized hyps of cardinality " << current_card
          << endl;
  }

  std::vector<partial_hyp_body>* sourceVectorPtr = &sourceVector;
  std::vector<partial_hyp_body>* targetVectorPtr = &targetVector;

  size_t sum_extensions = 0;

  misc::OFileStream best_hyps_out(best_hyp_filename);

  WRITE_BEST_HYPO(best_hyps_out.get(), 15, "card", "threshold", "histogram",
                  "mapping", "predecessor", "#hyps_total", "diff_best-worst");

  // main loop
  while (running) {
    LOG(INFO) << "beam search main loop" << " current_card = " << current_card;

    std::vector<partial_hyp_body>& sourceVectorRef = *sourceVectorPtr;
    std::vector<partial_hyp_body>& targetVectorRef = *targetVectorPtr;

    if (current_card >= card_max) {
      sum_extensions += current_hyps;
      LOG(INFO) << "#extensions: " << sum_extensions << endl;
      break;
    }

    word_id_t next_f = extOrder.getWordAt(current_card);

    dump_statistics();
    // double omp_wtime_start = 0.0;// omp_get_wtime();
    boost::atomic<size_t> stats_total_hyps(0);
    CHECK_LT(constraints.size(), 10);
    boost::atomic<size_t> stats_accepted_hyps[10];
    for (size_t i = 0; i < 10; ++i) stats_accepted_hyps[i] = 0;
    boost::atomic<size_t> output_hyp_no(0);

#pragma omp parallel  // shared(sourceVector, targetVector)
    {
#pragma omp master
      {

        std::string parallelization_str = "NO_OPENMP";
#ifdef WITH_OPENMP
        parallelization_str = std::to_string(omp_get_num_threads()) + " threads";
#endif

        LOG(INFO) << "expanding " << current_hyps << " hypotheses for cipher token '" 
          << f_vocab_.getWord(next_f) << "' and cardinality "
          << current_card << "/" << card_max << "' using " << parallelization_str;

        sum_extensions += current_hyps;
      }

#pragma omp for
      for (size_t i = 0; i < hypVector.size(); ++i) {
        hypVector[i].parent_id = INVALID_PARENT;
      }

// expand current hyps
//////////////////////
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < current_hyps; ++i) {
        // set up mapping
        Mapping baseMapping(f_vocab_.size(), e_vocab_.size());
        Mapping::hypToMapping(sourceVectorRef[i], &baseMapping, current_card);

        size_t local_target_hyp = output_hyp_no.fetch_add(1);

        // loop over all possible extensions
        for (size_t e = 0; e < e_vocab_.size(); ++e) {
          // static position
          // size_t local_target_hyp = i*e_vocab_.size()+e;

          // possibly implement dynamic expansion
          // size_t local_target_hyp = __sync_fetch_and_add(&...,1);
          partial_hyp_head& target = hypVector[local_target_hyp];
          stats_total_hyps++;
          // todo: move to constraints.hh
          bool accept = (sourceVectorRef[i].e_word_count[e] < limit_e[e]) &&
                        (special_e_words.count(e) == 0);
          // warning - check: some constraints may require to be ALWAYS exectued
          for (size_t j = 0; j < constraints.size() && accept; ++j) {
            stats_accepted_hyps[j]++;

            Constraint::ConstraintAcceptCode cur_accept =  constraints[j]->accept(e, next_f, baseMapping,
                                             sourceVectorRef[i], target);

            if (cur_accept == Constraint::REJECT) {
              accept = false;
              break;
            }
            if (cur_accept == Constraint::FORCE_KEEP) {
              accept = true;
              break;
            }
          }

          if (accept) {
            local_target_hyp = output_hyp_no.fetch_add(1);
            stats_accepted_hyps[constraints.size()]++;
            target.parent_id = i;
            assert(i < current_hyps);
            target.e = e;
            target.f = next_f;

            // calculate score
            int adding_code = baseMapping.add_(e, next_f);
            if (adding_code != 1) {
#pragma omp critical(print)
              {
                LOG(INFO) << "WARNING: ADDINGCODE != 1 FOR E='"
                      << e_vocab_.getWord(e) << "' (" << e << ")\tF='"
                      << f_vocab_.getWord(next_f) << "' (" << next_f << ")"
                      << endl;
                LOG(INFO) << baseMapping.dumpToString(e_vocab_, f_vocab_, &extOrder);
              }
            }
            assert(adding_code == 1);  // should be a consistent add

            // make vectorized updateScore method, calculating the different
            // expansions in one loop
            // reuse language model contexts

            target.score = 0;
            target.scores.resize(features.size(), 0);
            for (size_t j = 0; j < features.size(); ++j) {
              if (features[j]->active(current_card)) {
                target.scores[j] = sourceVectorRef[i].scores[j];
                features[j]->updateScore(baseMapping, next_f, target.scores[j]);
                target.score +=
                    features[j]->getWeight(next_f) * target.scores[j];
              } else {
                target.scores[j] = 0;
              }
            }

            // update correct count
            if ((referenceMapping != 0) &&
                referenceMapping->isCorrect(target.e, target.f)) {
              target.cor_count = sourceVectorRef[i].cor_count + 1;
            } else {
              target.cor_count = sourceVectorRef[i].cor_count;
            }

            baseMapping.removeMapping(e, next_f);

          } else {
            // invalidate
            // target.parent_id = INVALID_PARENT;
          }
        }
      }  // end omp parallel for
    }    // end omp parallel

    // double omp_wtime = 0.0;//omp_get_wtime()-omp_wtime_start;
    // LOG(INFO) << "~ " << hypVector.size() << " hyps, took " << omp_wtime
    // << "s = " << hypVector.size()/omp_wtime << " hyp/s" << endl;

    // omp_wtime_start = omp_get_wtime();

    // conservative estimate for highest hyp we touched, at max hypVector.size()
    size_t max_hyp = (current_hyps) * (e_vocab_.size());
    if (max_hyp > hypVector.size()) max_hyp = hypVector.size();

    // Histogram Pruning
    //        if (log_) {
    // ERROR
    // ERROR: The usage of pruneHist[current_card] is not correct here !!!!!!
    // ERROR
    LOG(INFO) << "only sorting " << output_hyp_no << " hyps vs prune histosize "
          << pruneHist[current_card] << "...";
#ifdef WITH_OPENMP
    __gnu_parallel::partial_sort(
        hypVector.begin(), hypVector.begin() + output_hyp_no,
        hypVector.begin() + output_hyp_no, cmp_partial_hyp_head);
#else
    std::partial_sort(hypVector.begin(), hypVector.begin() + output_hyp_no,
                      hypVector.begin() + output_hyp_no, cmp_partial_hyp_head);
#endif
    // LOG(INFO) << "took " << omp_get_wtime()-omp_wtime_start << "s" << endl;

    LOG(INFO) << "copying hyps in parallel" << endl;

    // omp_wtime_start = omp_get_wtime();

    size_t number_old_hyps = current_hyps;
    double bestScore = hypVector[0].score;
    std::vector<partial_hyp_head*> survived_heads;
    std::vector<size_t> e_histo(e_vocab_.size(), 0);
    std::vector<size_t> pred_histo(current_hyps, 0);

    size_t num_hyps_to_copy =
        std::min(pruneHist[current_card], output_hyp_no - 0);
    double used_prune_thres = pruneThresh[current_card];
    size_t used_prune_mapping = prune_mapping[current_card];
    size_t used_prune_predecessor = prune_predecessor[current_card];
    bool find_correct = true;
    word_id_t cheating_best_mapping_correct_count = 0;
    size_t cheating_best_mapping_idx = 0;

    CHECK_GT(num_hyps_to_copy, 0) << "need at least one hyp to copy";

    for (size_t i = 0;
         (i < hypVector.size()) && ((i < num_hyps_to_copy) || find_correct);
         ++i) {
      partial_hyp_head* hv_i = &(hypVector[i]);
      if (hv_i->parent_id == INVALID_PARENT) {
        break;
      }

      if (hv_i->cor_count > cheating_best_mapping_correct_count) {
        cheating_best_mapping_correct_count = hv_i->cor_count;
        cheating_best_mapping_idx = i;
      }

      bool criterium_histo = i < num_hyps_to_copy;
      bool criterium_thresh = hv_i->score >= (bestScore - used_prune_thres);
      bool criterium_min_hyp = (i < pruneMinHyp[current_card]);
      bool criterium_mapping = e_histo[hv_i->e] < used_prune_mapping;
      bool criterium_pred =
          pred_histo[hv_i->parent_id] < used_prune_predecessor;
      bool survive = criterium_histo &&
                     (criterium_thresh || criterium_min_hyp) &&
                     criterium_mapping && criterium_pred;
      if (survive) {
        survived_heads.push_back(hv_i);
        ++e_histo[hv_i->e];
        ++pred_histo[hv_i->parent_id];
      }
    }

    CHECK_GT(survived_heads.size(), 0) << "no heads survived";

    {
      partial_hyp_head* hv_i = &(hypVector[cheating_best_mapping_idx]);
      WRITE_BEST_HYPO(best_hyps_out.get(), 15, current_card,
                      bestScore - hv_i->score, cheating_best_mapping_idx + 1,
                      e_histo[hv_i->e], pred_histo[hv_i->parent_id],
                      current_hyps, bestScore - hv_i->score);

      LOG(INFO) << "best remaining hypothesis (cheating, since we use the reference for "
        << "this) has " << size_t(cheating_best_mapping_correct_count) << "/"
        << size_t(current_card + 1) << " correct decisions, cardinality " <<
        (current_card + 1);

      LOG(INFO) << "best remaining hypothesis (cheating, since we use the reference for "
        << "this) would survive with " << "threshold-pruning "
        << bestScore - hv_i->score << ", histogram-pruning " << (cheating_best_mapping_idx +1)
        << ", mapping-pruning " << e_histo[hv_i->e] << ", predecessor-pruning "
        << pred_histo[hv_i->parent_id];

      LOG(INFO) << "there were " << current_hyps << " hypotheses in total";
      LOG(INFO) << "scores range from " << bestScore << " to " << hv_i->score;
    }

    // todo: sometimes number_old_hyps < survived_heads ... ???
    LOG(INFO) << "pruned " << output_hyp_no
          << " hyps down to " << survived_heads.size() << " hyps (kept "
          << 100.0 * static_cast<double>(survived_heads.size()) /
                 static_cast<double>(output_hyp_no) << "%)";
// loop over survived hyp heads
//////////////////////
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < survived_heads.size(); ++i) {
      partial_hyp_head* hv_i = survived_heads[i];
      // ERROR
      // ERROR: The usage of pruneThresh[current_card] is not correct here !!!
      // ERROR
      if (hv_i->parent_id != INVALID_PARENT) {
        // update position for next hyp in targetVectorRef (threadsafe)

        partial_hyp_body& tv_i = targetVectorRef[i];

        assert(hv_i->parent_id < number_old_hyps);
        tv_i = sourceVectorRef[hv_i->parent_id];

        tv_i.word_pairs[current_card].e = hv_i->e;
        tv_i.word_pairs[current_card].f = hv_i->f;
        ++tv_i.e_word_count[hv_i->e];
        tv_i.score = hv_i->score;
        for (size_t j = 0; j < features.size(); ++j) {
          tv_i.scores[j] = hv_i->scores[j];
        }
        tv_i.parent_id = hv_i->parent_id;
        tv_i.cor_count = hv_i->cor_count;

        // output hyp
        if (printBest && i == 0) {
          Mapping mapping(f_vocab_.size(), e_vocab_.size());
          for (size_t j = 0; j < current_card + 1; ++j) {
            mapping.add_(tv_i.word_pairs[j].e, tv_i.word_pairs[j].f);
          }
          mapping.setScore(tv_i.score);
#pragma omp critical(print)
          {
            std::string marker_all_correct =
                (tv_i.cor_count == (current_card + 1)) ? "(all_correct (known via reference mapping))"
                                                     : "(not_correct (know via reference mapping))";
            LOG(INFO) << "# correct mappings of best scoring hypopthesis (known via reference mapping): " << size_t(tv_i.cor_count) << " "
                  << marker_all_correct << endl;
            LOG(INFO) << mapping.dumpToString(e_vocab_, f_vocab_, &extOrder);
          }
        }
      }
    }
    current_hyps = survived_heads.size();

    // LOG(INFO) << "copying/expanding survived hyps took "
    // << omp_get_wtime()-omp_wtime_start << "s" << endl;
    LOG(INFO) << "constraints stats: " << endl;
    LOG(INFO) << "constraint[x]: " << stats_accepted_hyps[0] << "/"
          << stats_total_hyps << " ("
          << double(stats_accepted_hyps[0]) / double(stats_total_hyps) * 100.0
          << "%) hyps" << endl;
    for (size_t i = 0; i < constraints.size(); ++i) {
      LOG(INFO) << "constraint[" << i << "]: " << stats_accepted_hyps[i + 1] << "/"
            << stats_total_hyps << " (" << double(stats_accepted_hyps[i + 1]) /
                                               double(stats_total_hyps) * 100.0
            << "%) hyps" << endl;
    }

    if ((option_searchlog_ == option_searchlog::LINEAR) ||
        (option_searchlog_ == option_searchlog::COMPACT)) {
      size_t run_to = 0;
      if (option_searchlog_ == option_searchlog::LINEAR) {
        run_to = current_hyps;
      } else if (option_searchlog_ == option_searchlog::COMPACT) {
        run_to = 1;
      }

      LOG(INFO) << "writing hyps to search log" << endl;

// dump debug info about current stack
#pragma omp for schedule(static, 1)
      for (size_t i = 0; i < run_to; ++i) {
        partial_hyp_body& tv_i = targetVectorRef[i];

        if (tv_i.word_pairs[current_card].e == 0) {
          LOG(WARNING) << "found hyp with e=<unk> : targetVectorRef[" << i
                << "]" << endl;
        }

        if (tv_i.word_pairs[current_card].f == 0) {
          LOG(WARNING) << "found hyp with f=<unk> : targetVectorRef[" << i
                << "]" << endl;
        }

        CHECK_NE(tv_i.word_pairs[current_card].e, 0);
        CHECK_NE(tv_i.word_pairs[current_card].f, 0);

        std::string marker = "";
        // read correct count if we have a reference mapping
        if (referenceMapping != 0) {
          if ((current_card + 1) == tv_i.cor_count) {
            marker = "\tALL_CORRECT";
          }
        }

#pragma omp critical
        {
          *search_log_os_ << std::setprecision(15) << current_card << "\t" << i;
          *search_log_os_ << "\t" << tv_i.parent_id << "\t"
                          << f_vocab_.getWord(tv_i.word_pairs[current_card].f);
          *search_log_os_ << "\t"
                          << e_vocab_.getWord(tv_i.word_pairs[current_card].e)
                          << "\t" << tv_i.score;
          for (size_t i = 0; i < features.size(); ++i) {
            *search_log_os_ << "\t" << tv_i.scores[i];
          }
          *search_log_os_ << "\t" << tv_i.cor_count << marker << endl;
        }
      }
    }

    assert(current_hyps > 0);
    std::swap<std::vector<partial_hyp_body>*>(sourceVectorPtr, targetVectorPtr);
    current_card++;
  }

  std::vector<partial_hyp_body>& sourceVectorRef = *sourceVectorPtr;

  assert(current_hyps <= sourceVectorRef.size());
  // not needed anymore since we are sorting fully anyway
  // LOG(INFO) << "performing final sort for best hyp on first " << current_hyps
  // << " of " << sourceVectorRef.size() << " hyps" << endl;
  // __gnu_parallel::sort(sourceVectorRef.begin(),
  // sourceVectorRef.begin() + current_hyps, cmp_partial_hyp_body);

  LOG(INFO) << "creating final mappings" << endl;
  std::vector<Mapping> result;
  for (size_t i = 0; i < current_hyps; ++i) {
    Mapping mapping(f_vocab_.size(), e_vocab_.size());
    for (size_t j = 0; j < current_card; ++j) {
      mapping.add_(sourceVectorRef[i].word_pairs[j].e,
                   sourceVectorRef[i].word_pairs[j].f);
      mapping.setScore(sourceVectorRef[i].score);
    }
    result.push_back(mapping);
  }

  LOG(INFO) << "created " << result.size() << " nodes" << endl;
  return result;
}

void BeamSearch::unsetRunning() { running = false; }
