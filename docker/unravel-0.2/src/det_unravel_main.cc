// todo: remove endl from logging statements
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(searchlog, "no", "possible options: linear|compact|no");
DEFINE_string(output_prefix, "output_prefix_", "prefix of output files");
DEFINE_uint64(extension_limit_e, 1, "e.g. for 1:1 ciphers select 1");
DEFINE_bool(print_best, false, "print statistics for best hypothesis");
DEFINE_string(ref_mapping_type, "ID", "ID|ID_");
DEFINE_string(ref_mapping_fn, "", "file with reference mapping");

DEFINE_string(lm, "", "target language model");

DEFINE_string(counts_1, "", "1gram source counts");
DEFINE_string(counts_2, "", "2gram source counts");
DEFINE_string(counts_3, "", "3gram source counts");
DEFINE_string(counts_4, "", "4gram source counts");
DEFINE_string(counts_5, "", "5gram source counts");
DEFINE_string(counts_6, "", "6gram source counts");
DEFINE_string(counts_7, "", "7gram source counts");
DEFINE_string(counts_8, "", "8gram source counts");

// todo: maybe remove ngram_feature, since multi_ngram_feature can provide the same functionality
DEFINE_uint64(ngram_feature_order, 0, "which order to start");
DEFINE_uint64(multi_ngram_feature_lowest_order, 0, "which order to start");

DEFINE_string(extorder_fn, "", "file with extension order");
DEFINE_string(extorder_type, "", "ngram|file");
DEFINE_uint64(extorder_beamsize, 100, "beamsize for extension order calculation");
DEFINE_double(extorder_counts_1_weight, 0.0, "weight for extension order calcualtion");
DEFINE_double(extorder_counts_2_weight, 0.0, "weight for extension order calcualtion");
DEFINE_double(extorder_counts_3_weight, 0.0, "weight for extension order calcualtion");
DEFINE_double(extorder_counts_4_weight, 0.0, "weight for extension order calcualtion");
DEFINE_double(extorder_counts_5_weight, 0.0, "weight for extension order calcualtion");
DEFINE_double(extorder_counts_6_weight, 0.0, "weight for extension order calcualtion");
DEFINE_double(extorder_counts_7_weight, 0.0, "weight for extension order calcualtion");
DEFINE_double(extorder_counts_8_weight, 0.0, "weight for extension order calcualtion");

DEFINE_uint64(pruning_range1_from, 0, "pruning range from");
DEFINE_uint64(pruning_range1_to, 999999, "pruning range to exclusive");
DEFINE_uint64(pruning_range1_histogram_min, 1, "pruning: minimal number of hypothesis");
DEFINE_uint64(pruning_range1_histogram_max, 100, "pruning: maximal number of hypothesis");
DEFINE_double(pruning_range1_threshold, 99999999.9, "pruning threshold");
DEFINE_uint64(pruning_range1_mapping_max, 999999, "pruning: maximal number of hypothesis with same local mapping");
DEFINE_uint64(pruning_range1_predecessor_max, 999999, "pruning: maximal number of hypothesis with same predecessor");

DEFINE_uint64(pruning_range2_from, 0, "pruning range from");
DEFINE_uint64(pruning_range2_to, 0, "pruning range to exclusive");
DEFINE_uint64(pruning_range2_histogram_min, 1, "pruning: minimal number of hypothesis");
DEFINE_uint64(pruning_range2_histogram_max, 100, "pruning: maximal number of hypothesis");
DEFINE_double(pruning_range2_threshold, 99999999.9, "pruning threshold");
DEFINE_uint64(pruning_range2_mapping_max, 999999, "pruning: maximal number of hypothesis with same local mapping");
DEFINE_uint64(pruning_range2_predecessor_max, 999999, "pruning: maximal number of hypothesis with same predecessor");

DEFINE_uint64(pruning_range3_from, 0, "pruning range from");
DEFINE_uint64(pruning_range3_to, 0, "pruning range to exclusive");
DEFINE_uint64(pruning_range3_histogram_min, 1, "pruning: minimal number of hypothesis");
DEFINE_uint64(pruning_range3_histogram_max, 100, "pruning: maximal number of hypothesis");
DEFINE_double(pruning_range3_threshold, 99999999.9, "pruning threshold");
DEFINE_uint64(pruning_range3_mapping_max, 999999, "pruning: maximal number of hypothesis with same local mapping");
DEFINE_uint64(pruning_range3_predecessor_max, 999999, "pruning: maximal number of hypothesis with same predecessor");

DEFINE_uint64(pruning_range4_from, 0, "pruning range from");
DEFINE_uint64(pruning_range4_to, 0, "pruning range to exclusive");
DEFINE_uint64(pruning_range4_histogram_min, 1, "pruning: minimal number of hypothesis");
DEFINE_uint64(pruning_range4_histogram_max, 100, "pruning: maximal number of hypothesis");
DEFINE_double(pruning_range4_threshold, 99999999.9, "pruning threshold");
DEFINE_uint64(pruning_range4_mapping_max, 999999, "pruning: maximal number of hypothesis with same local mapping");
DEFINE_uint64(pruning_range4_predecessor_max, 999999, "pruning: maximal number of hypothesis with same predecessor");

DEFINE_uint64(pruning_range5_from, 0, "pruning range from");
DEFINE_uint64(pruning_range5_to, 0, "pruning range to exclusive");
DEFINE_uint64(pruning_range5_histogram_min, 1, "pruning: minimal number of hypothesis");
DEFINE_uint64(pruning_range5_histogram_max, 100, "pruning: maximal number of hypothesis");
DEFINE_double(pruning_range5_threshold, 99999999.9, "pruning threshold");
DEFINE_uint64(pruning_range5_mapping_max, 999999, "pruning: maximal number of hypothesis with same local mapping");
DEFINE_uint64(pruning_range5_predecessor_max, 999999, "pruning: maximal number of hypothesis with same predecessor");


#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <csignal>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "beam_search.hh"
#include "global.hh"
#include "config_options.hh"
#include "constraint.hh"
#include "counts.hh"
#include "extension_order.hh"
#include "feature.hh"
#include "hypothesis_struct.hh"
#include "kenlm_with_vocab.hh"
#include "ngram_feature.hh"
#include "multi_ngram_feature.hh"
#include "reference_mapping.hh"
#include "misc.hh"

// can be removed
#include "resource.hh"
#include "cooc.hh"
#include "oracle_feature.hh"
#include "cooc_feature.hh"
#include "embedding_feature.hh"
#include "phrase_feature.hh"
#include "embedding.hh"


using std::cerr;
using std::endl;

std::ofstream used_config_file_out;
boost::iostreams::filtering_ostream used_config_stream_out;

BeamSearch* global_bs = 0;

void sighandler(int sig) {
  (void)sig;
  LOG(INFO) << "Keyboard Interrupt - Exiting";
  if (global_bs != 0) {
    global_bs->unsetRunning();
  } else {
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  INIT_MAIN("find closest sentences using a lexicon in two corpora\n");
  MISC_ENVINFO(MAX_LEN);
  MISC_ENVINFO(BS_MAX_ORDER);
  MISC_VARINFO(partial_hyp_body);
  MISC_VARINFO(partial_hyp_head);

  signal(SIGINT, &sighandler);
  check_global_options_integrity();

  LOG(INFO) << "using file prefix '" << FLAGS_output_prefix << "' for all subsequent file output";

  Vocab e_vocab, f_vocab;

  LOG(INFO) << "setting up LM";
  KenLmWithVocab lm(FLAGS_lm, &e_vocab);

  LOG(INFO) << "cache vocab types";
  e_vocab.cacheTypes();

  // special symbols that we need later on
  std::vector<std::string> special_words_pot = {
    e_vocab.getWord(e_vocab.sos()),
    e_vocab.getWord(e_vocab.eos()),
    e_vocab.getWord(e_vocab.unk()) };

  std::vector<Counts*> counts(9, nullptr);
  // todo: use new ...
  LOG(INFO) << "setting up counts";
  Counts counts_1(true);
  Counts counts_2(true);
  Counts counts_3(true);
  Counts counts_4(true);
  Counts counts_5(true);
  Counts counts_6(true);
  Counts counts_7(true);
  Counts counts_8(true);

  // use macro
  if (FLAGS_counts_1 != "") {
    counts_1.read(FLAGS_counts_1, 1, 0, f_vocab, true);
    counts[1] = &counts_1;
    // todo: use new
  }
  if (FLAGS_counts_2 != "") {
    counts_2.read(FLAGS_counts_2, 2, 0, f_vocab, true);
    counts[2] = &counts_2;
  }
  if (FLAGS_counts_3 != "") {
    counts_3.read(FLAGS_counts_3, 3, 0, f_vocab, true);
    counts[3] = &counts_3;
  }
  if (FLAGS_counts_4 != "") {
    counts_4.read(FLAGS_counts_4, 4, 0, f_vocab, true);
    counts[4] = &counts_4;
  }
  if (FLAGS_counts_5 != "") {
    counts_5.read(FLAGS_counts_5, 5, 0, f_vocab, true);
    counts[5] = &counts_5;
  }
  if (FLAGS_counts_6 != "") {
    counts_6.read(FLAGS_counts_6, 6, 0, f_vocab, true);
    counts[6] = &counts_6;
  }
  if (FLAGS_counts_7 != "") {
    counts_7.read(FLAGS_counts_7, 7, 0, f_vocab, true);
    counts[7] = &counts_7;
  }
  if (FLAGS_counts_8 != "") {
    counts_8.read(FLAGS_counts_8, 8, 0, f_vocab, true);
    counts[8] = &counts_8;
  }

  LOG(INFO) << "setting up reference mapping for evaluation";
  // todo: make lowercase
  ReferenceMapping referenceMapping;
  if (FLAGS_ref_mapping_type == "ID") {
    referenceMapping.id(e_vocab, f_vocab);
  } else if (FLAGS_ref_mapping_type == "ID_") {
    referenceMapping.idUnderscore(e_vocab, f_vocab);
  } else if (FLAGS_ref_mapping_fn != "") {
    misc::IFileStream referenceMappingStream(FLAGS_ref_mapping_fn);
    // todo: change ReferenceMapping to directly take filename as string
    referenceMapping.read(referenceMappingStream.get(), e_vocab, f_vocab);
  } else {
    LOG(FATAL) << "no reference mapping defined";
  }


  ////////////////////////////////////////////////////////////////////
  // todo: rename to potential_special_words (?)
  LOG(INFO) << "setting up extensionOrder";
  ExtensionOrder ext_order(f_vocab, special_words_pot);
  size_t ext_order_size = ext_order.size();
  LOG(INFO) << "ext_order: " << ext_order_size << " words in default ext-order.";

  if (FLAGS_extorder_fn != "") {
    ext_order.fillFromFile(f_vocab, FLAGS_extorder_fn);
    LOG(INFO) << "ext_order: filled " << ext_order.size() - ext_order_size
      << " words from file \"" << FLAGS_extorder_fn << "\"";
    ext_order_size = ext_order.size();
  } else if (FLAGS_extorder_type == "ngram") {
    std::vector<Counts*> count_vec;
    std::vector<double> ppl_vec;
    if ((FLAGS_counts_8 != "") && (FLAGS_extorder_counts_8_weight != 0.0)) {
      count_vec.push_back(&counts_8);
      ppl_vec.push_back(FLAGS_extorder_counts_8_weight);
    }
    if ((FLAGS_counts_7 != "") && (FLAGS_extorder_counts_7_weight != 0.0)) {
      count_vec.push_back(&counts_7);
      ppl_vec.push_back(FLAGS_extorder_counts_7_weight);
    }
    if ((FLAGS_counts_6 != "") && (FLAGS_extorder_counts_6_weight != 0.0)) {
      count_vec.push_back(&counts_6);
      ppl_vec.push_back(FLAGS_extorder_counts_6_weight);
    }
    if ((FLAGS_counts_5 != "") && (FLAGS_extorder_counts_5_weight != 0.0)) {
      count_vec.push_back(&counts_5);
      ppl_vec.push_back(FLAGS_extorder_counts_5_weight);
    }
    if ((FLAGS_counts_4 != "") && (FLAGS_extorder_counts_4_weight != 0.0)) {
      count_vec.push_back(&counts_4);
      ppl_vec.push_back(FLAGS_extorder_counts_4_weight);
    }
    if ((FLAGS_counts_3 != "") && (FLAGS_extorder_counts_3_weight != 0.0)) {
      count_vec.push_back(&counts_3);
      ppl_vec.push_back(FLAGS_extorder_counts_3_weight);
    }
    if ((FLAGS_counts_2 != "") && (FLAGS_extorder_counts_2_weight != 0.0)) {
      count_vec.push_back(&counts_2);
      ppl_vec.push_back(FLAGS_extorder_counts_2_weight);
    }
    if ((FLAGS_counts_1 != "") && (FLAGS_extorder_counts_1_weight != 0.0)) {
      count_vec.push_back(&counts_1);
      ppl_vec.push_back(FLAGS_extorder_counts_1_weight);
    }
    CHECK_GE(count_vec.size(), 1);
    ext_order.fillHighestNgramFreqOrderBeam(count_vec, ppl_vec,
        FLAGS_extorder_beamsize);

    ext_order.writeToFile(f_vocab, count_vec, FLAGS_output_prefix + "ext_order.gz");
  } else {
    LOG(FATAL) << "extension order is not configured";
  }

  LOG(INFO) << "ext_order: filled " << ext_order.size() - ext_order_size
            << " words dynamically";
  LOG(INFO) << "extension order entries: " << ext_order.size();
  LOG(INFO) << "f_vocab entries: " << f_vocab.size();
  CHECK_EQ(ext_order.size(), f_vocab.size());


  LOG(INFO) << "caching new ngrams";
  for (size_t i = 1; i <= 8; ++i) {
    if (counts[i] != nullptr) {
      (counts[i])->fillNewNgrams(ext_order);
    }
  }


  ////////////////////////////////////////////////////////////////////
  LOG(INFO) << "preparing features";
  std::vector<Feature*> features;
  NgramFeature ngram_feat(false);
  MultiNgramFeature multi_ngram_feat(false);
  if (FLAGS_ngram_feature_order != 0) {
    CHECK(counts[FLAGS_ngram_feature_order] != 0);
    LOG(INFO) << "activating ngram feature";
    ngram_feat.activate(*counts[FLAGS_ngram_feature_order], lm,
        FLAGS_ngram_feature_order, ext_order, 1.0);
    features.push_back(&ngram_feat);
  }
  if (FLAGS_multi_ngram_feature_lowest_order != 0) {
    CHECK(counts[FLAGS_multi_ngram_feature_lowest_order] !=  0);
    for (size_t i = 8; i >= FLAGS_multi_ngram_feature_lowest_order; --i) {
      if (counts[i] != nullptr) {
        multi_ngram_feat.addCounts(*counts[i]);
      }
    }
    LOG(INFO) << "activating multi_ngram feature";
    multi_ngram_feat.activate(lm, ext_order, 1.0);
    features.push_back(&multi_ngram_feat);
  } else {
      LOG(FATAL) << "No feature selected";
  }
  LOG(INFO) << "prepared features";


  // this needs to be fixed
  ////////////////////////////////////////////////////////////////////
  BeamSearch bs(f_vocab, e_vocab, special_words_pot);

  LOG(INFO) << "reading search log configuration" << std::endl;
  std::string searchlog_option = FLAGS_searchlog;
  if ((searchlog_option != "linear") && (searchlog_option != "compact") &&
      (searchlog_option != "no")) {
    LOG(FATAL) << "Error: searchlog has unknown value \"" << searchlog_option
              << "\"";
  }
  std::ofstream searchlog_file_out;

  // todo: use new code for compressed output
  boost::iostreams::filtering_ostream searchlog_stream_out;
  if ((searchlog_option == "linear") || (searchlog_option == "compact")) {
    searchlog_file_out.open(FLAGS_output_prefix + "search_log.gz",
                            std::ios_base::out | std::ios_base::binary);
    searchlog_stream_out.push(boost::iostreams::gzip_compressor());
    searchlog_stream_out.push(searchlog_file_out);
    if (searchlog_option == "linear") {
      bs.setSearchLog(option_searchlog::LINEAR, &searchlog_stream_out);
    } else if (searchlog_option == "compact") {
      bs.setSearchLog(option_searchlog::COMPACT, &searchlog_stream_out);
    }
  }


  ////////////////////////////////////////////////////////////////////
  LOG(INFO) << "reading extension limits configuration";
  int main_limit_e = FLAGS_extension_limit_e;
  std::vector<size_t> limit_e(e_vocab.size(), main_limit_e);

  ////////////////////////////////////////////////////////////////////
  LOG(INFO) << "reading pruning configuration";

  // make one struct 
  // {
  //   size_t, ... score_t
  // }
  std::vector<size_t> pruneHistoMax(f_vocab.size(), 1);
  std::vector<size_t> pruneHistoMin(f_vocab.size(), 1);
  std::vector<score_t> pruneThresh(f_vocab.size(), 0);
  std::vector<size_t> pruneMapping(f_vocab.size(), 1);
  std::vector<size_t> prunePredecessor(f_vocab.size(), 1);

  size_t from, to_excl;
  from = std::max(FLAGS_pruning_range2_from, static_cast<google::uint64>(0));
  to_excl = std::min(FLAGS_pruning_range1_to,
      static_cast<google::uint64>(f_vocab.size()));
  for (size_t j = from; j < to_excl; ++j) {
    pruneHistoMax[j] = FLAGS_pruning_range1_histogram_max;
    pruneHistoMin[j] = FLAGS_pruning_range1_histogram_min;
    pruneThresh[j] = FLAGS_pruning_range1_threshold;
    pruneMapping[j] = FLAGS_pruning_range1_mapping_max;
    prunePredecessor[j] = FLAGS_pruning_range1_predecessor_max;
  }
  from = std::max(FLAGS_pruning_range2_from, static_cast<google::uint64>(0));
  to_excl = std::min(FLAGS_pruning_range2_to,
    static_cast<google::uint64>(f_vocab.size()));
  for (size_t j = from; j < to_excl; ++j) {
    pruneHistoMax[j] = FLAGS_pruning_range2_histogram_max;
    pruneHistoMin[j] = FLAGS_pruning_range2_histogram_min;
    pruneThresh[j] = FLAGS_pruning_range2_threshold;
    pruneMapping[j] = FLAGS_pruning_range2_mapping_max;
    prunePredecessor[j] = FLAGS_pruning_range2_predecessor_max;
  }
  from = std::max(FLAGS_pruning_range3_from, static_cast<google::uint64>(0));
  to_excl = std::min(FLAGS_pruning_range3_to,
    static_cast<google::uint64>(f_vocab.size()));
  for (size_t j = from; j < to_excl; ++j) {
    pruneHistoMax[j] = FLAGS_pruning_range3_histogram_max;
    pruneHistoMin[j] = FLAGS_pruning_range3_histogram_min;
    pruneThresh[j] = FLAGS_pruning_range3_threshold;
    pruneMapping[j] = FLAGS_pruning_range3_mapping_max;
    prunePredecessor[j] = FLAGS_pruning_range3_predecessor_max;
  }
  from = std::max(FLAGS_pruning_range4_from, static_cast<google::uint64>(0));
  to_excl = std::min(FLAGS_pruning_range4_to,
      static_cast<google::uint64>(f_vocab.size()));
  for (size_t j = from; j < to_excl; ++j) {
    pruneHistoMax[j] = FLAGS_pruning_range4_histogram_max;
    pruneHistoMin[j] = FLAGS_pruning_range4_histogram_min;
    pruneThresh[j] = FLAGS_pruning_range4_threshold;
    pruneMapping[j] = FLAGS_pruning_range4_mapping_max;
    prunePredecessor[j] = FLAGS_pruning_range4_predecessor_max;
  }
  from = std::max(FLAGS_pruning_range5_from, static_cast<google::uint64>(0));
  to_excl = std::min(FLAGS_pruning_range5_to,
      static_cast<google::uint64>(f_vocab.size()));
  for (size_t j = from; j < to_excl; ++j) {
    pruneHistoMax[j] = FLAGS_pruning_range5_histogram_max;
    pruneHistoMin[j] = FLAGS_pruning_range5_histogram_min;
    pruneThresh[j] = FLAGS_pruning_range5_threshold;
    pruneMapping[j] = FLAGS_pruning_range5_mapping_max;
    prunePredecessor[j] = FLAGS_pruning_range5_predecessor_max;
  }


  ////////////////////////////////////////////////////////////////////
  std::vector<Constraint*> constraints;

  // debugging options
  bool print_best = FLAGS_print_best;

  global_bs = &bs;
  std::vector<Mapping> final_hypothesis = bs.search(
      ext_order, limit_e, constraints, features, &referenceMapping, pruneThresh,
      pruneHistoMin, pruneHistoMax, pruneMapping, prunePredecessor,
      FLAGS_output_prefix + "best_hyps.gz", print_best);

  ///////////////////////////////////////////// output mapping
  misc::OFileStream ostr_map(FLAGS_output_prefix + "result_mapping.gz");
  final_hypothesis[0].writeLexicon(e_vocab, f_vocab, &ostr_map.get());

  ///////////////////////////////////////////// output ratio correct wrong
  misc::OFileStream ostr_ratio(FLAGS_output_prefix + "final_eval.gz");
  final_hypothesis[0]
      .printRatioCorrectWrong(&referenceMapping, &ostr_ratio.get());

  return 0;
}
