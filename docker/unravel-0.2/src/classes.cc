#include "classes.hh"
#include "global.hh"
#include "ngram.hh"
#include "misc.hh"

#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/progress.hpp>

Classes::Classes() : vocab_(0) {}

class_id_t Classes::getClass(word_id_t word_id) const {
  if (word_id >= class_map_.size()) {
    std::cerr << "CANNOT FIND CLASS FOR '" << vocab_->getWord(word_id) << "' ("
              << word_id << ")" << std::endl;
  }

  CHECK_LT(word_id, class_map_.size());
  return class_map_[word_id];
}

void Classes::read(boost::iostreams::filtering_istream& classes_is,
                   Vocab& vocab, bool verbose) {
  max_class_ = 0;
  vocab_ = &vocab;
  size_t line_count = 0;
  size_t errors = 0;
  std::cerr << "Reading classes";

  for (std::string current_line; std::getline(classes_is, current_line);) {
    if (line_count % 10000 == 0) std::cerr << ".";

    std::vector<std::string> fields;
    boost::split(fields, current_line, boost::is_any_of("\t "));
    CHECK_GE(fields.size(), 2);
    std::string word = fields[0];
    size_t word_class = atoi(fields[1].c_str());
    word_id_t word_id = vocab_->getId(word);

    if (word_id == VOCAB_WORD_NOT_FOUND) {
      if (verbose || errors < 5)
        std::cerr << "skipping class for word " << word << std::endl;
      errors++;
      continue;
    }

    if (word_id >= class_map_.size()) class_map_.resize(word_id + 1, 0);
    if (word_class > max_class_) max_class_ = word_class;

    class_map_[word_id] = word_class;
    ++line_count;
  }

  if (!verbose && errors > 0)
    std::cerr << "..." << std::endl << errors
              << " errors while reading (skipped over most of them because of "
                 "verbosity settings)" << std::endl;

  std::cerr << std::endl;
  std::cerr << "read " << line_count
            << " lines resulted in vocab_size=" << vocab_->size()
            << " with highest class=" << max_class_ << std::endl;

  CHECK_GT(vocab_->size(), 0);
}

Classes::~Classes() {}

Vocab& Classes::getVocab() { return *vocab_; }

const Vocab& Classes::getVocabConst() const { return *vocab_; }
