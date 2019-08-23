#include "counts.hh"
#include "global.hh"
#include "ngram.hh"
#include "misc.hh"

#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

using std::cerr;
using std::endl;

Counts::Counts(bool pad_sent_start)
    : vocab_(0), zerogram_count_(0), pad_sent_start_(pad_sent_start) {}

void Counts::addSentenceStartCounts(const std::vector<word_id_t>& vec,
                                    double count,
                                    std::map<Ngram, count_t>& ngrams_map) {

  for (size_t act_start = 2; act_start < order_; ++act_start) {
    Ngram ngram(order_);
    ngram.count = count;
    for (size_t i = 0; i < act_start; ++i) {
      ngram.t[i] = vocab_->getIdFail("<s>");
    }
    for (size_t i = act_start; i < order_; ++i) {
      ngram.t[i] = vec[i - act_start + 1];
    }
    zerogram_count_ += count;
    // add to map
    if (ngrams_map.find(ngram) != ngrams_map.end())
      ngrams_map[ngram] += count;
    else
      ngrams_map[ngram] = count;
  }
}

void Counts::read(std::string filename, order_t order, count_t mincount,
                  Vocab& vocab, bool calc_word_to_ngrams) {
  misc::IFileStream ifs(filename);
  order_ = order;
  vocab_ = &vocab;
  size_t line_count = 0;
  size_t total_bytes = misc::getStreamSize(ifs.getIfstream());
  LOG(INFO) << "reading counts file '" << filename << "' with size "
            << misc::intToHumanStr(total_bytes) << "b";
  std::map<Ngram, count_t> ngrams_map;  // not really nice

  size_t bytes_read = 0;
  long bytes_since_last = 0;
  for (std::string current_line; std::getline(ifs.get(), current_line);) {
    ++line_count;
    bytes_read += (static_cast<long>(ifs.getIfstream().tellg()) -
                 static_cast<long>(bytes_since_last));
    bytes_since_last = ifs.getIfstream().tellg();
    VLOG_EVERY_N(3, 10000) << "reading line " << line_count;

    std::vector<std::string> fields;
    boost::split(fields, current_line, boost::is_any_of("\t"));

    CHECK_GE(fields.size(), 2) << "expecting 2 columns per line";

    // read double
    std::istringstream count_stream(fields[1]);
    double count = 0;
    count_stream >> count;

    if (count < mincount) continue;

    // add code for keeping track if we already read that ngram
    std::vector<word_id_t> vec = vocab_->getIdsAdd(fields[0]);
    if (vec.size() != order_) {
      LOG(INFO) << "SKIPPING NGRAM '" << current_line
                << "' BECAUSE OF MISMATCHING ORDER " << vec.size() << " vs "
                << order_;
      continue;
    } else {
      if (pad_sent_start_ && (vec[0] == vocab_->getId("<s>"))) {
        addSentenceStartCounts(vec, count, ngrams_map);
      }
      Ngram ngram(order_);
      ngram.count = count;
      for (size_t i = 0; i < order_; ++i) {
        ngram.t[i] = vec[i];
      }
      zerogram_count_ += ngram.count;
      // add to map
      if (ngrams_map.find(ngram) != ngrams_map.end())
        ngrams_map[ngram] += count;
      else
        ngrams_map[ngram] = count;
    }
  }

  for (std::map<Ngram, count_t>::const_iterator it = ngrams_map.begin();
       it != ngrams_map.end(); ++it) {
    Ngram ngram = it->first;
    ngram.count = it->second;
    ngrams_.push_back(ngram);
  }

  size_t bytes = ngrams_.size() * sizeof(Ngram);
  LOG(INFO) << "done. got " << line_count << " lines, " << ngrams_map.size() << " entries, order=" << static_cast<int>(order_) << " with vocab.size=" << vocab_->size() << " using " << misc::intToHumanStr(bytes) <<"b";

  CHECK_GT(ngrams_.size(), 0);
  CHECK_GT(order_, 0);
  CHECK_GT(vocab_->size(), 0);

  // todo: move to separate method
  if (calc_word_to_ngrams) {
    LOG(INFO) << "setup lookup table: 'word' -> 'n-grams containing that word' for '" << filename << "'";
    word_to_ngrams_.resize(vocab_->size());

    for (size_t i = 0; i < ngrams_.size(); ++i) {
      for (size_t j = 0; j < order_; ++j) {
        word_to_ngrams_[ngrams_[i].t[j]].insert(&ngrams_[i]);
      }
    }
  }
}

void Counts::fillNewNgrams(const ExtensionOrder& ext_order) {
  LOG(INFO) << "calculating fully mapped (n=" << static_cast<int>(order_) <<")-grams for extension order";
  CHECK_GT(vocab_->size(), 0);

  // initialize
  newNgrams_.resize(vocab_->size());
  //  newBigrams.resize(cipher_.getVocab().size());
  std::vector<bool> wordsCalculated(vocab_->size(), false);

  // loop over extension order
  for (size_t i = 0; i < ext_order.size(); ++i) {

    word_id_t newWord = ext_order.getWordAt(i);
    wordsCalculated[newWord] = true;
    const std::set<Ngram*>& cur_ngrams = word_to_ngrams_[newWord];

    // loop over all affected ngrams
    for (std::set<Ngram*>::iterator ngramIt = cur_ngrams.begin();
         ngramIt != cur_ngrams.end(); ++ngramIt) {
      Ngram& cur_ngram = *(*ngramIt);
      bool ngramComplete = true;
      // check if ngram is completed
      for (order_t i = order_; i > 0; --i) {
        if ((!(wordsCalculated[cur_ngram.t[i - 1]])) &&
            (word_id_t(cur_ngram.t[i - 1]) != newWord)) {
          ngramComplete = false;
          break;
        }
      }
      if (ngramComplete) {
        newNgrams_[newWord].insert(*ngramIt);
      }
    }
  }
}

Counts::~Counts() {}

Vocab& Counts::getVocab() { return *vocab_; }

const Vocab& Counts::getVocabConst() const { return *vocab_; }

const std::vector<Ngram>& Counts::getNgrams() const { return ngrams_; }

order_t Counts::getOrder() const { return order_; }

const std::vector<std::set<Ngram*> >& Counts::getWordToNgrams() const {
  return word_to_ngrams_;
}

const std::vector<std::set<Ngram*> >& Counts::getNewNgrams() const {
  return newNgrams_;
}

void Counts::printNumberOfNgrams() const {
  LOG(INFO) << "vocab size: " << vocab_->size();
  LOG(INFO) << "#diff ngrams: " << ngrams_.size();
  LOG(INFO) << "zerogram count: " << zerogram_count_;
}

void Counts::printCounts() const {
  for (size_t i = 0; i < ngrams_.size(); ++i) {
    for (size_t j = 0; j < order_; ++j) {
      cerr << vocab_->getWord(ngrams_[i].t[j]) << " ";
    }
    cerr << "\t" << ngrams_[i].count << endl;
  }
}
