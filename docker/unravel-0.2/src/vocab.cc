#include "vocab.hh"

#include <boost/algorithm/string.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <limits>

#include "global.hh"
#include "misc.hh"
#include "murmur_hash.h"

#include <fst/symbol-table.h>
#include <glog/logging.h>

using std::cerr;
using std::endl;

/** constructs an empty Vocab - filling must be done separately */
Vocab::Vocab()
    : word_vec_(0), cache_valid_(false), unk_(0), sos_(0), eos_(0), null_(0) {
  // always need this special Word:
  // especially <unk> must be always mapped to 0
  addWord("<unk>");
  // faster lookups
  word_map_.max_load_factor(0.1);
}

Vocab::~Vocab() {}

/** makes new Word available */
word_id_t Vocab::addWord(const std::string& word) {
  invalidateCache();
  uint64_t hash = MurmurHash64(word.c_str(), word.length());
  const auto& it = word_map_.find(hash);
  if (it != word_map_.end()) {
    return it->second;
  } else {
    word_id_t id = word_vec_.size();
    word_map_[hash] = id;
    word_vec_.push_back(word);
    return id;
  }
}

bool Vocab::containsId(word_id_t word_id) const {
  if (word_id >= word_vec_.size()) return false;
  return word_vec_[word_id] != "";
}

bool Vocab::containsWord(const std::string& word) const {
  uint64_t hash = MurmurHash64(word.c_str(), word.length());
  const auto& it = word_map_.find(hash);
  return (it != word_map_.end());
}

/** returns the int representation of the given string representation */
word_id_t Vocab::getId(const std::string& word) const {
  if (word_vec_.size() > 6 && word_vec_[6] == word) return 6;
  if (word_vec_.size() > 7 && word_vec_[7] == word) return 7;
  if (word_vec_.size() > 8 && word_vec_[8] == word) return 8;
  if (word_vec_.size() > 9 && word_vec_[9] == word) return 9;
  if (word_vec_.size() > 10 && word_vec_[10] == word) return 10;
  if (word_vec_.size() > 11 && word_vec_[11] == word) return 11;
  if (word_vec_.size() > 12 && word_vec_[12] == word) return 12;
  if (word_vec_.size() > 13 && word_vec_[13] == word) return 13;
  if (word_vec_.size() > 14 && word_vec_[14] == word) return 14;

  uint64_t hash = MurmurHash64(word.c_str(), word.length());
  const auto& it = word_map_.find(hash);
  if (it != word_map_.end()) {
    return it->second;
  } else {
    return VOCAB_WORD_NOT_FOUND;
  }
}

/** returns the int representation of the given string representation */
word_id_t Vocab::getIdFail(const std::string& word) const {
  uint64_t hash = MurmurHash64(word.c_str(), word.length());
  const auto& it = word_map_.find(hash);
  if (it != word_map_.end()) {
    return it->second;
  } else {
    LOG(FATAL) << "failed to get word_id for word '" << word
               << "' but expected it to be a known word in vocab "
               << "with size=" << size();
  }
  return 0;
}

/** returns the string representation of the given int representation */
std::string Vocab::getWord(word_id_t wordId) const {
  CHECK_LT(wordId, word_vec_.size());
  return word_vec_[wordId];
}

/** Converts string to int vector representation
 *  if a Word is not known until now, fail */
std::vector<word_id_t> Vocab::getIdsFail(const std::string& text) {
  std::vector<word_id_t> vec;
  vec.reserve(10);
  std::stringstream streamText(text);
  while (!streamText.eof()) {
    std::string sym;
    std::getline(streamText, sym, ' ');
    if (!sym.empty()) {
      vec.push_back(getIdFail(sym));
    }
  }
  return vec;
}

/** Converts string to int vector representation
 *  if a Word is not known until now it is added to the vocab */
std::vector<word_id_t> Vocab::getIdsAdd(const std::string& text) {
  std::vector<word_id_t> vec;
  vec.reserve(10);
  std::stringstream streamText(text);
  while (!streamText.eof()) {
    std::string sym;
    std::getline(streamText, sym, ' ');
    if (!sym.empty()) {
      addWord(sym);
      vec.push_back(getIdFail(sym));
    }
  }
  return vec;
}

/** Converts string to int vector representation
 *  if a Word is not known by the vocab, its in representation is 0 */
std::vector<word_id_t> Vocab::getIdsMark(const std::string& text) const {
  std::vector<word_id_t> vec;
  std::stringstream streamText(text);
  while (!streamText.eof()) {
    std::string sym;
    std::getline(streamText, sym, ' ');
    if (!sym.empty()) {
      vec.push_back(getId(sym));
    }
  }
  return vec;
}

/** Converts int vector to string representation */
std::string Vocab::getWords(const std::vector<word_id_t>& vec) const {
  std::stringstream streamText;
  for (auto it = vec.begin(); it != vec.end(); ++it) {
    streamText << getWord(*it);
    if (it + 1 != vec.end()) streamText << " ";
  }
  return streamText.str();
}

void Vocab::print(std::ostream& os) const {
  os << "Vocab::print" << endl;
  os << "vec_size: " << word_vec_.size() << endl;
  os << "map_size: " << word_map_.size() << endl;
  os << "map_load_factor: " << word_map_.load_factor() << endl;
  os << "map_bucket_count: " << word_map_.bucket_count() << endl;
  for (size_t i = 0; i < word_vec_.size(); ++i) {
    os << i << "\t'" << word_vec_[i] << "'" << endl;
  }
}

size_t Vocab::size() const { return word_vec_.size(); }

#ifdef WITH_OPENFST
fst::SymbolTable* Vocab::getFstSymbolTable() const {
  fst::SymbolTable* stable = new fst::SymbolTable();
  for (word_id_t i = 0; i < word_vec_.size(); ++i) {
    stable->AddSymbol(getWord(i), i);
  }
  return stable;
}
#endif

#ifdef WITH_OPENFST
void Vocab::fromFstSymbolTable(const fst::SymbolTable* table) {
  CHECK_GE(std::numeric_limits<word_id_t>::max(), table->NumSymbols());
  for (word_id_t i = 0; i < table->NumSymbols(); ++i) {
    std::string word = table->Find(i);
    setWord_(i, word);
  }
}
#endif

void Vocab::setWord_(word_id_t word_id, const std::string& word) {
  invalidateCache();
  uint64_t hash = MurmurHash64(word.c_str(), word.length());
  word_map_[hash] = word_id;
  if (word_id >= word_vec_.size()) {
    word_vec_.resize(word_id + 1, "VOCAB_WORD_NOT_FOUND");
  }
  word_vec_[word_id] = word;
}

void Vocab::clear() {
  word_map_.clear();
  word_vec_.clear();
}

void Vocab::write(boost::iostreams::filtering_ostream& os) const {
  LOG(INFO) << "writing vocab of size " << word_vec_.size() << endl;
  for (word_id_t i = 0; i < word_vec_.size(); ++i) {
    os << int(i) << "\t" << word_vec_[i] << endl;
  }
}

void Vocab::read(const std::string& fn, bool read_ids) {
  invalidateCache();
  LOG(INFO) << "reading vocab from '" << fn
            << "' (current vocab size=" << size() << ")";
  misc::IFileStream istr(fn);
  size_t num_entries = 0;
  for (std::string current_line; std::getline(istr.get(), current_line);) {
    std::vector<std::string> fields;
    boost::split(fields, current_line, boost::is_any_of(" \t"));
    if (read_ids) {
      CHECK_EQ(2, fields.size());
      word_id_t word_id = atoi(fields[0].c_str());
      std::string word = fields[1];
      CHECK(containsWord(word) == false);
      CHECK(containsId(word_id) == false);
      setWord_(word_id, word);
      ++num_entries;
    } else {
      std::string word = fields[0];
      // if (!containsWord(word)) LOG(WARNING) << "word '" << word
      //                                   << "' already in vocab";
      addWord(word);
      ++num_entries;
    }
  }

  LOG(INFO) << "vocab reading done (" << num_entries
            << " entries read, current vocab size=" << size() << ")";
}

std::vector<std::vector<word_id_t>> Vocab::readCorpusAdd(
    const std::string& fn) {
  invalidateCache();
  std::vector<std::vector<word_id_t>> result;
  LOG(INFO) << "read corpus from '" << fn << "'";
  misc::IFileStream ifs(fn);

  for (std::string str; std::getline(ifs.get(), str);) {
    boost::algorithm::trim(str);
    result.push_back(getIdsAdd(str + " </s>"));
  }

  LOG(INFO) << result.size() << " lines" << endl;
  return result;
}

void Vocab::cacheTypes() {
  // normal words
  types_.resize(word_vec_.size(), VOCAB_TYPE_NORMAL);
  normal_words_.clear();
  for (size_t i = 0; i < word_vec_.size(); ++i) {
    std::string word = word_vec_[i];
    if (word == "<s>") {
      types_[i] = VOCAB_TYPE_SOS;
      sos_ = i;
    } else if (word == "</s>") {
      types_[i] = VOCAB_TYPE_EOS;
      eos_ = i;
    } else if (word == "<unk>") {
      types_[i] = VOCAB_TYPE_UNK;
      unk_ = i;
    } else if (word == "<null>") {
      types_[i] = VOCAB_TYPE_NULL;
      null_ = i;
    } else {
      normal_words_.push_back(i);
      types_[i] = VOCAB_TYPE_NORMAL;
    }
  }
  cache_valid_ = true;
}

word_type_t Vocab::getType(word_id_t w) const {
  CHECK(cache_valid_);
  return types_[w];
}

bool Vocab::isNormalOrNull(word_id_t w) const {
  CHECK(cache_valid_);
  return (types_[w] == VOCAB_TYPE_NORMAL || types_[w] == VOCAB_TYPE_NULL);
}

bool Vocab::isNormal(word_id_t w) const {
  CHECK(cache_valid_);
  return (types_[w] == VOCAB_TYPE_NORMAL);
}

bool Vocab::isNull(word_id_t w) const {
  CHECK(cache_valid_);
  return (types_[w] == VOCAB_TYPE_NULL);
}

size_t Vocab::normal_plus_null_size() const {
  CHECK(cache_valid_);
  if (null_ == 0) {
    return getNormalWords().size();
  } else {
    return getNormalWords().size() + 1;
  }
}

const std::vector<word_id_t>& Vocab::getNormalWords() const {
  CHECK(cache_valid_);
  return normal_words_;
}
