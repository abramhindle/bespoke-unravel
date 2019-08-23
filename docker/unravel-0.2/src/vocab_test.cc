#include "vocab.hh"
#include <gtest/gtest.h>

TEST(VocabTestSuite, VocabTestInsert) {
  Vocab vocab;

  ASSERT_EQ(0, vocab.getId("<unk>")) << "word should be unknown";
  ASSERT_EQ("<unk>", vocab.getWord(0)) << "word with id 0 should be '<unk>'";

  ASSERT_EQ(0, vocab.getId("first")) << "word should be unknown";
  ASSERT_EQ(0, vocab.getId("first")) << "do not add word when asking for it";
  ASSERT_EQ(1, vocab.addWord("first")) << "first new word should have id 1";
  ASSERT_EQ(1, vocab.addWord("first")) << "first new word should have id 1";
  ASSERT_EQ(1, vocab.getId("first")) << "first new word should have id 1";
  ASSERT_EQ("first", vocab.getWord(1)) << "word with id 1 should be 'first'";

  vocab.clear();
  EXPECT_EQ(0, vocab.getId("first")) << "word should be unknown";
}

TEST(VocabTestSuite, VocabTestStringEquality) {
  Vocab vocab;

  // id=0 is reserved for <unk>
  for (int i = 1; i < 100; ++i) {
    std::string word = std::to_string((long long)i);
    ASSERT_EQ(i, vocab.addWord(word)) << "ith word index != i";
  }

  for (int i = 1; i < 100; ++i) {
    std::string word = std::to_string((long long)i);
    ASSERT_EQ(i, vocab.getId(word)) << "ith word index != i";
  }
}

TEST(VocabTestSuite, VocabTestReadFileUnique) {
  Vocab vocab;
  std::string vocab_file = misc::make_file(
      "a\n"
      "b\n"
      "c\n"
      "d\n");
  vocab.read(vocab_file, false);
  // + 1 = <unk>
  ASSERT_EQ(5, vocab.size());
}

TEST(VocabTestSuite, VocabTestReadFileNonUnique) {
  Vocab vocab;
  std::string vocab_file = misc::make_file("a\n");
  vocab.read(vocab_file, false);
  // + 1 = <unk>
  ASSERT_EQ(2, vocab.size());
}

TEST(VocabTestSuite, VocabTestReadFileIds) {
  Vocab vocab;
  std::string vocab_file = misc::make_file(
      "1 a\n"
      "2 b\n"
      "3 c\n"
      "4 d\n");
  vocab.read(vocab_file, true);
  // + 1 = <unk>
  ASSERT_EQ(5, vocab.size());
  ASSERT_EQ(1, vocab.getId("a"));

  std::string vocab_file2 = misc::make_file(
      "5 e\n"
      "6 f\n"
      "7 g\n"
      "8 h\n");
  vocab.read(vocab_file2, true);
  // + 1 = <unk>
  ASSERT_EQ(9, vocab.size());
  ASSERT_EQ(8, vocab.getId("h"));
}
