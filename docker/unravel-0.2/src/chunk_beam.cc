#include <glog/logging.h>

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
#include <map>
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
#include "lexicon.hh"
#include "fst_permute.hh"
#include "fst_beamsearch.hh"
#include "chunk_beam.hh"

std::vector<word_id_t> getWindow(size_t center, size_t window_size,
                                 const std::vector<word_id_t> &vec,
                                 word_id_t null_left, word_id_t null_right) {
  std::vector<word_id_t> res;

  size_t left = std::max(0, int(center) - int(window_size));

  // fill left range with 0
  for (size_t i = center; i < window_size; ++i) res.push_back(null_left);

  size_t right = std::min(vec.size(), center + window_size + 1);
  for (size_t i = left; i < right; ++i) res.push_back(vec[i]);

  // fill right range with 0
  for (size_t i = right; i < center + window_size + 1; ++i)
    res.push_back(null_right);

  return res;
}

void ChunkBeam::addChunks(const std::vector<std::vector<word_id_t>> &sent,
                          word_id_t null_left, word_id_t null_right,
                          std::map<std::vector<word_id_t>, size_t> *chunks) {
  LOG(INFO) << "init chunk beam sent.size=" << sent.size() << endl;
  // loop over all sentences
  for (size_t idx = 0; idx < sent.size(); ++idx) {
    for (size_t center = 0; center < sent[idx].size(); ++center) {
      const std::vector<word_id_t> &chunk =
          getWindow(center, windowSize_, sent[idx], null_left, null_right);
      // LOG(INFO) << "center=" << center << " " << eVocab_.getWords(sent[idx])
      // <<
      // endl;
      // LOG(INFO) << eVocab_.getWords(chunk) << endl;
      ++(*chunks)[chunk];
    }
  }
  LOG(INFO) << "initialized " << chunks->size() << " chunks" << endl;
}

std::set<word_id_t> ChunkBeam::getCandidates(const std::vector<word_id_t> &input_sentence) const {
  // todo: to be implemented
  return std::set<word_id_t>();
}
