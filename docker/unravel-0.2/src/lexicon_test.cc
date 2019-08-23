#include <gtest/gtest.h>
#include "vocab.hh"
#include "lexicon_sparse.hh"

Vocab InitVocab() {
  Vocab vocab;
  vocab.addWord("a");  // id=1
  vocab.addWord("b");  // id=2
  vocab.addWord("c");  // id=3
  vocab.addWord("d");  // id=4
  vocab.cacheTypes();
  return vocab;
}


TEST(LexiconTestSuite, LexiconCompare) {

}


TEST(LexiconTestSuite, LexiconTestInsert) {
  Vocab eVocab = InitVocab();
  Vocab fVocab = InitVocab();

  size_t threads = 2;
  size_t beamsize = 1;
  size_t minobs = 0;

  LexiconSparse lex(&eVocab, &fVocab, beamsize, 1.00);
  lex.init();

  LOG(INFO) << "empty lexicon";
  double cur_lambda = 0.0;
  while (cur_lambda <= 1.0) {
    CHECK_EQ(lex.lin_score_fast(1,1), 0.0);
    CHECK_EQ(lex.score(1,1), - std::numeric_limits<double>::infinity());
    //CHECK_LT(std::abs(lex.score(1,1) - ::log(fVocab.size()-3)), 0.001);
    cur_lambda += 0.05;
  }

  LOG(INFO) << "add counts";
  lex.add(eVocab.getId("a"),fVocab.getId("a"),3.0);
  lex.add(eVocab.getId("a"),fVocab.getId("a"),1.0);
  lex.learn();

  LOG(INFO) << "query all";
  std::vector<double> scores;

  for (const auto & e : eVocab.getNormalWords()) {
    for (const auto & f : fVocab.getNormalWords()) {
      score_t score = lex.score(f,e);
      scores.push_back(score);
      LOG(INFO) << eVocab.getWord(e) << " " << fVocab.getWord(f) << " " << score;
    }
  }

  LOG(INFO) << "add first";
  lex.add(eVocab.getId("a"),fVocab.getId("a"),1.0);
  lex.learn();

  LOG(INFO) << "compare all";
  size_t i=0;
  for (const auto & e : eVocab.getNormalWords()) {
    for (const auto & f : fVocab.getNormalWords()) {
      score_t score = lex.score(f,e);
      CHECK_EQ(score, scores[i++]);
    }
  }




}
