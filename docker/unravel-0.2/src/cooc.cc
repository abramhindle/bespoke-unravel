#include "cooc.hh"
#include "global.hh"
#include "ngram.hh"
#include "misc.hh"

#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/progress.hpp>

Cooc::Cooc() : vocab_(0) {}

void Cooc::addCooc(word_id_t e, word_id_t e_prime) {
  cooc_map_[e].insert(e_prime);
  cooc_map_[e_prime].insert(e);
}

const std::set<word_id_t>* Cooc::getCooc(word_id_t word_id) const {
  std::map<word_id_t, std::set<word_id_t> >::const_iterator it =
      cooc_map_.find(word_id);
  if (it != cooc_map_.end()) return &(it->second);
  return 0;
}

void Cooc::read(boost::iostreams::filtering_istream& cooc_is, Vocab& vocab,
                bool verbose) {
  max_class_ = 0;
  vocab_ = &vocab;
  size_t line_count = 0;
  size_t errors = 0;
  LOG(INFO) << "Reading cooc" << std::endl;

  for (std::string current_line; std::getline(cooc_is, current_line);) {
    if (line_count % 10000 == 0) LOG(INFO) << "read " << line_count << "lines" << std::endl;

    std::vector<std::string> fields;
    boost::split(fields, current_line, boost::is_any_of("\t "));
    CHECK_GE(fields.size(), 2);
    std::string word = fields[0];
    word_id_t word_id = vocab_->getId(word);

    if (word_id == VOCAB_WORD_NOT_FOUND) {
      if (verbose || errors < 5)
        LOG(INFO) << "skipping cooc for word " << word << std::endl;
      errors++;
      continue;
    }

    for (size_t i = 1; i < fields.size(); ++i) {
      std::string cooc_word = fields[i];
      word_id_t cooc_word_id = vocab_->getId(cooc_word);
      if (cooc_word_id == VOCAB_WORD_NOT_FOUND) {
        if (verbose || errors < 5)
          LOG(WARNING) << "skipping cooc for word " << word << std::endl;
        errors++;
        continue;
      }

      addCooc(word_id, cooc_word_id);
    }

    ++line_count;
  }

  if (!verbose && errors > 0)
    LOG(WARNING) << "..." << std::endl << errors
              << " errors while reading (skipped over most of them because of "
                 "verbosity settings)" << std::endl;

   LOG(INFO) << "read " << line_count
            << " lines resulted in vocab_size=" << vocab_->size() << std::endl;

  CHECK_GT(vocab_->size(), 0);
}

Cooc::~Cooc() {}

Vocab& Cooc::getVocab() { return *vocab_; }

const Vocab& Cooc::getVocabConst() const { return *vocab_; }
