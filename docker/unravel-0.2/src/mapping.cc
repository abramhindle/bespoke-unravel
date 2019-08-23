#include "mapping.hh"

#include <string>
#include <iostream>
#include <sstream>

#include "misc.hh"

using std::cerr;
using std::endl;

void Mapping::hypToMapping(const partial_hyp_body &source_hyp,
                           Mapping *target_mapping, size_t card) {
  target_mapping->reset();
  target_mapping->setScore(source_hyp.score);
  for (size_t j = 0; j < card; ++j) {
    size_t e = source_hyp.word_pairs[j].e;
    // check if word_pairs and e_word_count are consistent
    if (source_hyp.e_word_count[e] == 0) {
#pragma omp critical(print)
      {
        LOG(FATAL) << "ERROR IN source_hyp.sympairs[" << j << "].e: "
                   << "WORD E=" << e << " "  //" (" << e_vocab_.getWord(e) << ") "
                   << "BUT source_hyp.e_word_count[" << e << "]=0" << endl;
      }
    }
    CHECK_GT(source_hyp.e_word_count[e], 0);
    target_mapping->add_(source_hyp.word_pairs[j].e,
                         source_hyp.word_pairs[j].f);
  }
}

Mapping::Mapping(size_t vocab_size_f, size_t vocab_size_e)
    : countMapE(0),
      mappingFToE(vocab_size_f, MAPPING_NOT_MAPPED),
      countE(vocab_size_e, 0),
      cardinality(0),
      score(0) {}

Mapping::~Mapping() {}

size_t Mapping::correctCount(const Mapping &reference) const {
  size_t correctCount = 0;
  for (unsigned int i = 0; i < mappingFToE.size(); ++i) {
    if ((mappingFToE[i] != MAPPING_NOT_MAPPED) &&
        (mappingFToE[i] == reference.getE(i)))
      ++correctCount;
  }
  return correctCount;
}

void Mapping::printRatioCorrectWrong(const ReferenceMapping *reference,
                                     std::ostream *os) const {
  if (reference == 0) {
    LOG(INFO) << "Could not calculate ratio correct / wrong, because no reference"
          << " mapping is given" << endl;
    return;
  }
  int correctCount = 0;
  for (word_id_t f = 0; f < mappingFToE.size(); ++f) {
    if (mappingFToE[f] != MAPPING_NOT_MAPPED) {
      if (reference->isCorrect(mappingFToE[f], f)) {
        ++correctCount;
      }
    }
  }
  (*os) << "score: " << getScore() << "\ttotal: " << getCardinality()
        << "\tcorrect: " << correctCount
        << "\tratio: " << correctCount * 1.0 / getCardinality() << endl;
}

void Mapping::dumpInfo(Vocab const &eVocab, Vocab const &fVocab,
                       std::ostream *os, bool compact) const {
  *os << cardinality << "\t" << getScore() << "\t";
  bool first = true;
  for (unsigned int i = 0; i < mappingFToE.size(); ++i) {
    if (mappingFToE[i] != MAPPING_NOT_MAPPED) {
      if (!first) *os << " ";
      first = false;
      if (!compact)
        *os << fVocab.getWord(i) << ":" << eVocab.getWord(mappingFToE[i]);
      else
        *os << eVocab.getWord(mappingFToE[i]);
    }
  }
  *os << endl;
}

void Mapping::dumpInfo(Vocab const &eVocab, Vocab const &fVocab,
                       boost::iostreams::filtering_ostream *os,
                       bool compact) const {
  (*os) << cardinality << "\t" << getScore() << "\t";
  bool first = true;
  for (unsigned int i = 0; i < mappingFToE.size(); ++i) {
    if (mappingFToE[i] != MAPPING_NOT_MAPPED) {
      if (!first) (*os) << " ";
      first = false;
      if (!compact)
        (*os) << fVocab.getWord(i) << ":" << eVocab.getWord(mappingFToE[i]);
      else
        (*os) << eVocab.getWord(mappingFToE[i]);
    }
  }
  (*os) << endl;
}

// other order e-f
void Mapping::writeLexicon(Vocab const &eVocab, Vocab const &fVocab,
                           boost::iostreams::filtering_ostream *os) const {
  //  *os << " e \t f \t weight" << endl;
  //  *os << "---\t---\t--------" << endl;
  for (unsigned int i = 0; i < mappingFToE.size(); ++i) {
    *os << fVocab.getWord(i) << "\t" << eVocab.getWord(mappingFToE[i])
        << "\t1.0" << endl;
  }
}

void Mapping::readLexicon(Vocab const &eVocab, Vocab const &fVocab,
                          boost::iostreams::filtering_istream *is) {
  for (std::string current_line; std::getline(*is, current_line);) {
    std::vector<std::string> fields;
    boost::split(fields, current_line, boost::is_any_of(" \t"));
    CHECK_GE(fields.size(), 2);
    std::string e = fields[0];
    std::string f = fields[1];
    word_id_t e_id = eVocab.getId(e);
    word_id_t f_id = fVocab.getId(f);
    add_(e_id, f_id);
  }
}

void Mapping::printInfo() const {
  LOG(INFO) << "Score: " << getScore() << " {";
  bool first = true;
  for (unsigned int i = 0; i < mappingFToE.size(); ++i) {
    if (mappingFToE[i] != MAPPING_NOT_MAPPED) {
      if (first) {
        cerr << i << ":" << mappingFToE[i];
        first = false;
      } else {
        cerr << ", " << i << ":" << mappingFToE[i];
      }
    }
  }
  cerr << "}" << endl;
}

std::string Mapping::dumpToString(Vocab const &eVocab, Vocab const &fVocab,
                           const ExtensionOrder *extOrder) const {
  std::string mapping = "{";
  for (unsigned int i = 0; i < mappingFToE.size(); ++i) {
    CHECK_NOTNULL(extOrder);
    word_id_t f = extOrder->getWordAt(i);
    if (mappingFToE[f] != MAPPING_NOT_MAPPED) {
      std::string fString = fVocab.getWord(f);
      std::string eString = eVocab.getWord(mappingFToE[f]);
      mapping += fString + ":" + eString + ", ";
    }
  }
  return "mapping { score=" + std::to_string(getScore()) + ", decision=" + mapping + " } ";
}

void Mapping::printInfo(Vocab const &eVocab, Vocab const &fVocab,
                        const Mapping &reference) const {
  bool firstCorrect = true;
  bool firstWrong = true;
  std::string correctMapping = "{";
  std::string wrongMapping = "\t\t{";
  int correctCount = 0;
  for (unsigned int i = 0; i < mappingFToE.size(); ++i) {
    if (mappingFToE[i] != MAPPING_NOT_MAPPED) {
      std::string eString = eVocab.getWord(mappingFToE[i]);
      std::string fString = fVocab.getWord(i);
      if (mappingFToE[i] == reference.getE(i)) {
        ++correctCount;
        if (firstCorrect) {
          correctMapping += fString + ":" + eString;
          firstCorrect = false;
        } else {
          correctMapping += ", " + fString + ":" + eString;
        }
      } else {
        if (firstWrong) {
          wrongMapping += fString + ":" + eString;
          firstWrong = false;
        } else {
          wrongMapping += ", " + fString + ":" + eString;
        }
      }
    }
  }
  std::stringstream sstream;
  sstream << correctCount << "," << cardinality - correctCount << " ";

  std::cout << "++++++++++++++++++++++++++++++ Hypothesis Info +++++++++++++++"
               "+++++++++++++++" << endl;
  std::cout << "Score: " << getScore() << "\t";
  std::cout << sstream.str() + correctMapping + "} " + wrongMapping + "}"
            << endl;
  //  std::cout<<estimatedBigramScoreFixedFirst+estimatedBigramScoreFixedSecond
  // +estimatedBigramScoreFixedNon << " " <<estimatedBigramScoreFixedFirst
  // <<" "<<estimatedBigramScoreFixedSecond<<" "
  // <<estimatedBigramScoreFixedNon<<endl;
  std::cout << "--------------------------------------------------------------"
               "---------------" << endl;
}

void Mapping::setScore(score_t score) { this->score = score; }

double Mapping::getScore(void) const { return score; }

int Mapping::getCardinality() const { return cardinality; }

std::vector<word_id_t> Mapping::applyMapping(
    std::vector<word_id_t> cipherVector) const {
  size_t length = cipherVector.size();
  std::vector<word_id_t> plainVector(length, VOCAB_WORD_NOT_FOUND);
  for (size_t i = 0; i < length; ++i) {
    plainVector[i] = mappingFToE[cipherVector[i]];
  }
  return plainVector;
}

bool operator<(Mapping const &lhs, Mapping const &rhs) {
  return lhs.getScore() < rhs.getScore();
}

#ifdef WITH_OPENFST
void Mapping::fstFgivenE(fst::VectorFst<fst::StdArc> *lexFst,
                         const Vocab &eVocab, const Vocab &fVocab,
                         score_t lambda) {
  lexFst->DeleteStates();
  lexFst->AddState();
  lexFst->SetStart(0);
  lexFst->SetFinal(0, 0.0);

  std::vector<word_id_t> eSpecial;
  std::vector<word_id_t> fSpecial;

  lexFst->AddArc(
      0, fst::StdArc(fVocab.getId("<s>"), eVocab.getId("<s>"), score, 0));
  fSpecial.push_back(fVocab.getId("<s>"));
  eSpecial.push_back(eVocab.getId("<s>"));
  lexFst->AddArc(
      0, fst::StdArc(fVocab.getId("</s>"), eVocab.getId("</s>"), score, 0));
  fSpecial.push_back(fVocab.getId("</s>"));
  eSpecial.push_back(eVocab.getId("</s>"));
  lexFst->AddArc(
      0, fst::StdArc(fVocab.getId("<unk>"), eVocab.getId("<unk>"), score, 0));
  fSpecial.push_back(fVocab.getId("<unk>"));
  eSpecial.push_back(eVocab.getId("<unk>"));
  lexFst->AddArc(
      0, fst::StdArc(fVocab.getId("<eps>"), eVocab.getId("<eps>"), score, 0));
  fSpecial.push_back(fVocab.getId("<eps>"));
  eSpecial.push_back(eVocab.getId("<eps>"));

  // skip special symbols
  score_t background = -::log(lambda / score_t(fVocab.size() - 3.0));

  for (word_id_t e = 0; e < eVocab.size(); ++e) {
    if (std::find(eSpecial.begin(), eSpecial.end(), e) != eSpecial.end())
      continue;

    size_t count = countE[e];
    score_t foreground = background;
    if (count > 0)
      foreground = -::log(exp(-background) + (1.0 - lambda) / score_t(count));

    for (word_id_t f = 0; f < fVocab.size(); ++f) {
      if (std::find(fSpecial.begin(), fSpecial.end(), f) != fSpecial.end())
        continue;
      if (count > 0 && mappingFToE[f] == e)
        score = foreground;
      else
        score = background;
      lexFst->AddArc(0, fst::StdArc(f, e, score, 0));
    }
  }
}
#endif
