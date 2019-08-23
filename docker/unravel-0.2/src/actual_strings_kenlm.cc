#include "actual_strings_kenlm.hh"

ActualStringsKenLm::ActualStringsKenLm(Vocab& vocab) : vocab(vocab) {}

ActualStringsKenLm::~ActualStringsKenLm() {}

void ActualStringsKenLm::Add(lm::WordIndex index, const StringPiece& str) {
  std::string word = str.as_string();
  vocab.addWord(word);
  // check whether calls to Add(...) had been in the expected manner
  CHECK_EQ(vocab.getId(word), index);
  CHECK_EQ(vocab.getWord(index), word);
}

Vocab ActualStringsKenLm::getVocab() const { return vocab; }
