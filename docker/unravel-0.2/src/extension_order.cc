#include "extension_order.hh"

#include <boost/progress.hpp>
#include <boost/lexical_cast.hpp>

#include <iomanip>
//#include <parallel/algorithm>

ExtensionOrder::ExtensionOrder(Vocab& f_vocab,
                               std::vector<std::string>& special_words)
    : special_words(special_words) {
  // for fast lookup
  std::set<word_id_t> f_special_words;

  // words to calculate
  words_calculated.resize(f_vocab.size(), false);
  for (word_id_t i = 0; i < f_vocab.size(); ++i) {
    words_to_calculate.insert(i);
  }

  // move special words to beginning
  LOG(INFO) << "removing special words " << special_words.size() << "words from extension order";
  for (word_id_t i = 0; i < special_words.size(); ++i) {
    std::string cur_word = special_words[i];
    word_id_t f_word_id = f_vocab.getId(cur_word);
    if (f_special_words.count(f_word_id) == 0) {
      f_special_words.insert(f_word_id);
      ext_order.push_back(f_word_id);
      words_to_calculate.erase(f_word_id);
      words_calculated[f_word_id] = true;
    }
  }
}

ExtensionOrder::~ExtensionOrder() {}

void ExtensionOrder::writeToFile(const Vocab &f_vocab,
    const std::vector<Counts*>& base_countss, const std::string &fn) const {

  misc::OFileStream outFile(fn);
  auto & outFileStream = outFile.get();

  std::vector<count_t> ngramsFixed(base_countss.size(), 0);
  std::vector<std::vector<size_t> > ngramsFixedHisto(
      base_countss.size(), std::vector<size_t>(ext_order.size(), 0));
  std::vector<std::vector<size_t> > ngramsNewlyFixedHisto(
      base_countss.size(), std::vector<size_t>(ext_order.size(), 0));

  outFileStream << "# extension order" << std::endl;
  outFileStream << "# size = " << ext_order.size() << std::endl;
  outFileStream << "# special symbols size = " << special_words.size() << std::endl;

  // calculate statistics
  for (size_t j = 0; j < base_countss.size(); ++j) {
    std::vector<bool> words_calculated(base_countss[0]->getVocabConst().size(),
                                       false);
    for (size_t i = 0; i < ext_order.size(); ++i) {
      words_calculated[ext_order[i]] = true;
      ngramsNewlyFixedHisto[j][i] = calcNumberOfAdditionalNgrams(
          *base_countss[j], words_calculated, ext_order[i]);
      ngramsFixed[j] += ngramsNewlyFixedHisto[j][i];
      ngramsFixedHisto[j][i] = ngramsFixed[j];
    }
  }

  outFileStream << "# fields are:" << std::endl;
  outFileStream << "#" << std::endl;
  outFileStream << "# * " << "rank" << std::endl;
  outFileStream << "# * " << "token" << std::endl;
  for (size_t j = 0; j < base_countss.size(); ++j) {
    outFileStream << "# * number of newly fixed " + boost::lexical_cast<std::string>(
        (int)base_countss[j]->getOrder()) + "grams" << std::endl;
  }
  outFileStream << std::endl;

  for (size_t i = 0; i < ext_order.size(); ++i) {
    outFileStream << i << "\t" << base_countss[0]->getVocabConst().getWord(ext_order[i]) << "\t";
    for (size_t j = 0; j < base_countss.size(); ++j) {
      size_t total_ngrams = ngramsFixed[j];
      if (i >= special_words.size()) {
        outFileStream << "\t" << ngramsNewlyFixedHisto[j][i];
      }
    }
    outFileStream << std::endl;
  }
}

void ExtensionOrder::fillFromFile(Vocab& f_vocab, const std::string &fn) {

  LOG(INFO) << "read extension order from file '" << fn << "'";
  misc::IFileStream orderFile(fn);
  CHECK_GT(f_vocab.size(), 0);

  LOG(INFO) << "f_vocab_size = " << f_vocab.size() << std::endl;

  size_t lines = 0;
  std::string currentWordLine;
  while (std::getline(orderFile.get(), currentWordLine)) {
    LOG(INFO) << "reading line: " << currentWordLine;

    if (currentWordLine.substr(0, 1) == "#") continue;
    ++lines;
    std::vector<std::string> fields;
    boost::split(fields, currentWordLine, boost::is_any_of("\t"));

    CHECK_GE(fields.size(), 2);

    std::string currentWord = fields[1];
    // ignoring fields[0]
    word_id_t fWordId = f_vocab.getId(currentWord);

    if (fWordId == 0) {
      LOG(WARNING) << "unknown word '" << currentWord << "' in ext_order file on line " << lines;
      continue;
    }

    if(words_calculated[fWordId]) {
      LOG(WARNING) << "word in extension order file was already covered for word '" << f_vocab.getWord(fWordId) << "'";
      continue;
    }

    ext_order.push_back(fWordId);
    words_to_calculate.erase(fWordId);
    words_calculated[fWordId] = true;
  }

  LOG(INFO) << "read " << lines << " lines from extension order file";
  LOG(INFO) << "extension order entries after file is read: "
            << ext_order.size();
  LOG(INFO) << "vocab entries: " << f_vocab.size();
}

// compare vectors
bool ExtensionOrder::compareGreater(std::vector<size_t>& big,
                                    std::vector<size_t>& small) {
  CHECK_EQ(small.size(), big.size());
  for (size_t i = 0; i < big.size(); ++i) {
    if (big[i] > small[i]) {
      return true;
    } else if (big[i] < small[i]) {
      return false;
    }
  }
  return false;
}

// beam search
void ExtensionOrder::fillHighestNgramFreqOrderBeam(
    std::vector<Counts*>& base_countss, std::vector<double> ppl_vec,
    size_t beam_size) {
  ExtOrderHyp* initHyp =
      new ExtOrderHyp(ext_order, words_calculated, base_countss.size());
  std::vector<ExtOrderHyp*> sourceVec;
  std::vector<ExtOrderHyp*> targetVec;
  sourceVec.push_back(initHyp);
  while (sourceVec[0]->ext_order.size() <
         base_countss[0]->getVocabConst().size()) {
    for (size_t i = 0; i < sourceVec.size(); ++i) {
      ExtOrderHyp* baseHyp = sourceVec[i];
      for (word_id_t w = 0; w < base_countss[0]->getVocabConst().size(); ++w) {
        if (!baseHyp->words_calculated[w]) {
          ExtOrderHyp* extendedHyp = new ExtOrderHyp(*baseHyp);
          size_t cur_score_pos = extendedHyp->scores[0].size() - 1;
          for (size_t c = 0; c < base_countss.size(); ++c) {
            extendedHyp->scores[c].push_back(
                extendedHyp->scores[c][cur_score_pos] +
                calcNumberOfAdditionalNgrams(*(base_countss[c]),
                                             extendedHyp->words_calculated, w));
          }
          extendedHyp->ext_order.push_back(w);
          extendedHyp->words_calculated[w] = true;
          targetVec.push_back(extendedHyp);
        }
      }
      delete baseHyp;
    }
    size_t used_beam_size = beam_size;
    //        if (sourceVec[0]->ext_order.size()>10) {
    //            used_beam_size = 1;
    //        } else {
    //            used_beam_size = beam_size;
    //        }
    size_t max_elems = std::min(used_beam_size, targetVec.size());
    std::partial_sort(targetVec.begin(), targetVec.begin() + max_elems,
                      targetVec.end(), doCompare(ppl_vec));
    sourceVec.resize(0);
    for (size_t i = 0; i < max_elems; ++i) {
      sourceVec.push_back(targetVec[i]);
    }
    for (size_t i = max_elems; i < targetVec.size(); ++i) {
      delete targetVec[i];
    }
    targetVec.resize(0);
  }
  ext_order = sourceVec[0]->ext_order;
  words_calculated = sourceVec[0]->words_calculated;
  words_to_calculate.clear();

  for (size_t i = 0; i < sourceVec.size(); ++i) {
    delete sourceVec[i];
  }
}

count_t ExtensionOrder::calcNumberOfAdditionalNgrams(
    const Counts& base_counts, const std::vector<bool>& wordsCalculated,
    word_id_t newWord, bool unique) const {
  count_t numberNewNgrams = 0;
  for (std::set<Ngram*>::iterator ngramIt =
           base_counts.getWordToNgrams()[newWord].begin();
       ngramIt != base_counts.getWordToNgrams()[newWord].end(); ++ngramIt) {
    bool ngramComplete = true;
    for (order_t i = 0; i < base_counts.getOrder(); ++i) {
      if ((!(wordsCalculated[(*ngramIt)->t[i]])) &&
          (!((word_id_t)(*ngramIt)->t[i] == newWord))) {
        ngramComplete = false;
        break;
      }
    }
    if (ngramComplete) {
      numberNewNgrams += unique ? 1 : (*ngramIt)->count;
    }
  }
  return numberNewNgrams;
}

word_id_t ExtensionOrder::getWordAt(word_id_t pos) const {
  CHECK_LT(pos, ext_order.size());
  return ext_order[pos];
}

word_id_t ExtensionOrder::size() const { return ext_order.size(); }


void ExtensionOrder::fillEmbedding(Vocab& f_vocab, Embedding* embedding, size_t beam_size) {
  LOG(FATAL) << "not yet implemented";
  //ExtOrderHyp* initHyp =
  //    new ExtOrderHyp(ext_order, words_calculated, base_countss.size());
  //std::vector<ExtOrderHyp*> sourceVec;
  //std::vector<ExtOrderHyp*> targetVec;
  //sourceVec.push_back(initHyp);
  ext_order.clear();
  for(size_t i=0;i<f_vocab.size();++i) ext_order.push_back(i);
}
