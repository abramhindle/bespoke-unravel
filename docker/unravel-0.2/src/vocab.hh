#ifndef VOCAB_H_
#define VOCAB_H_

#include <boost/iostreams/filtering_stream.hpp>
#include <map>
#include <unordered_map>
#include <ostream>
#include <string>
#include <vector>
//#include <google/sparse_hash_map>

#include "global.hh"
#include "misc.hh"
#include "misc_io.hh"

#ifdef WITH_OPENFST
namespace fst {
class SymbolTable;
}
#endif

// ToDo: Make small and check for bugs and runtime
#define DEFAULT_VOCAB_SIZE 27
#define VOCAB_WORD_NOT_FOUND 0

#define VOCAB_TYPE_NORMAL 0
#define VOCAB_TYPE_UNK 1
#define VOCAB_TYPE_SOS 2
#define VOCAB_TYPE_EOS 3
#define VOCAB_TYPE_NULL 4

/** The main task of this class is to provide functionality to establish an
 * one-to-one mapping between complex Words (strings of one or more letters)
 * and ints. In Application the int representation should be used where the
 * time consuming computations are done.
 */
class Vocab {
 public:
  Vocab();
  word_id_t addWord(const std::string&);
  std::string getWord(word_id_t wordId) const;
  bool containsWord(const std::string& word) const;
  bool containsId(word_id_t word_id) const;
  word_id_t getId(const std::string& word) const;
  word_id_t getIdFail(const std::string& word) const;
  virtual ~Vocab();
  std::vector<word_id_t> getIdsAdd(const std::string&);
  std::vector<word_id_t> getIdsFail(const std::string&);
  std::vector<word_id_t> getIdsMark(const std::string&) const;
  std::string getWords(const std::vector<word_id_t>&) const;
  void print(std::ostream& os) const;
  size_t size() const;
#ifdef WITH_OPENFST
  fst::SymbolTable* getFstSymbolTable() const;
  void fromFstSymbolTable(const fst::SymbolTable* table);
#endif
  void setWord_(word_id_t word_id, const std::string& word);
  void write(boost::iostreams::filtering_ostream& os) const;
  void read(const std::string& fn, bool read_ids);
  std::vector<std::vector<word_id_t>> readCorpusAdd(const std::string& fn);
  void clear();
  void cacheTypes();
  const std::vector<word_id_t>& getNormalWords() const;
  word_type_t getType(word_id_t w) const;
  bool isNormalOrNull(word_id_t w) const;
  bool isNormal(word_id_t w) const;
  bool isNull(word_id_t w) const;
  size_t normal_plus_null_size() const;

  void invalidateCache() {
    if (cache_valid_) {
      LOG(WARNING) << "invalidating cached vocab types";
    }
    cache_valid_ = false;
  }

  // unk() is always mapped to 0
  inline word_id_t unk() const {
    CHECK(cache_valid_);
    return unk_;
  }
  inline word_id_t sos() const {
    CHECK_NE(sos_, 0) << "sos token not known in vocab with size=" << size();
    return sos_;
  }
  inline word_id_t eos() const {
    CHECK_NE(eos_, 0) << "eos token not known in vocab with size=" << size();
    return eos_;
  }
  inline word_id_t null() const {
    CHECK_NE(null_, 0) << "null token not known in vocab with size=" << size();
    return null_;
  }
  inline bool has_null() const { return !(null_ == 0); }

 private:
  // google::sparse_hash_map<uint64_t, word_id_t> word_map_;
  std::unordered_map<uint64_t, word_id_t> word_map_;
  std::vector<std::string> word_vec_;
  std::vector<word_type_t> types_;
  std::vector<word_id_t> normal_words_;
  word_id_t unk_, sos_, eos_, null_;  // <unk> <s> </s> <null>
  bool cache_valid_;
};

#endif /* VOCAB_H_ */
