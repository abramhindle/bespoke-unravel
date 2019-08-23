#include "reference_mapping.hh"
#include <glog/logging.h>

#include <string>
#include <iostream>
#include <sstream>

using std::endl;
using std::cerr;

ReferenceMapping::ReferenceMapping() {}

ReferenceMapping::~ReferenceMapping() {}

void ReferenceMapping::id(const Vocab& evocab, const Vocab& fvocab) {
  for (size_t f = 0; f < fvocab.size(); ++f) {
    std::string f_string = fvocab.getWord(f);
    // remove '_' and index number
    size_t e = evocab.getId(f_string);
    if ((e == VOCAB_WORD_NOT_FOUND) && (f != VOCAB_WORD_NOT_FOUND)) {
      LOG(WARNING) << "ERROR WHEN SETTING UP IDENTITY "
        << "REFERENCE MAPPING BECAUSE AUF WORD '" << f_string << "'"
        << endl;
    } else {
      possibleEs_[f].insert(e);
    }
  }
}

void ReferenceMapping::idUnderscore(const Vocab& evocab, const Vocab& fvocab) {
  for (size_t f = 0; f < fvocab.size(); ++f) {
    std::string f_string = fvocab.getWord(f);
    size_t pos = f_string.find_last_of('_');
    f_string = f_string.substr(0, pos);

    size_t e = evocab.getId(f_string);

    if ((e == VOCAB_WORD_NOT_FOUND) && (f != VOCAB_WORD_NOT_FOUND)) {
      LOG(WARNING) << "ERROR WHEN SETTING UP IDENTITY "
        << "REFERENCE MAPPING BECAUSE AUF WORD '" << f_string << "'"
        << endl;
    } else {
      possibleEs_[f].insert(e);
    }
  }
}

void ReferenceMapping::read(boost::iostreams::filtering_istream& is,
                            const Vocab& evocab, const Vocab& fvocab) {
  size_t line_count = 0;
  size_t entries = 0;
  LOG(INFO) << "reading reference mapping";

  CHECK_GT(fvocab.size(), 0);
  CHECK_GT(evocab.size(), 0);
  LOG(INFO) << "eVocab.size = " << evocab.size() << ", "
            << "fVocab.size = " << fvocab.size();

  for (std::string current_line; std::getline(is, current_line);) {
    ++line_count;
    if (line_count % 10000 == 0) LOG(INFO) << "read " << line_count << " lines" << std::endl;
    std::vector<std::string> fields;
    boost::split(fields, current_line, boost::is_any_of("\t "));
    CHECK_GE(fields.size(), 2);
    std::string fstring = fields[0];
    std::string estring = fields[1];

    // try to find WORD, WORD_1, WORD_2, ...
    for (size_t i = 0; i < 5; ++i) {
      if (i > 0) fstring += "_" + std::to_string(static_cast<long long>(i));
      word_id_t f = fvocab.getId(fstring);
      if (f == VOCAB_WORD_NOT_FOUND) continue;
      for (size_t j = 0; j < 5; ++j) {
        if (j > 0) estring += "_" + std::to_string(static_cast<long long>(j));
        word_id_t e = evocab.getId(estring);
        if (e == VOCAB_WORD_NOT_FOUND) continue;
        VLOG(2) << "adding f=" << fstring << " e=" << estring;
        possibleEs_[f].insert(e);
        entries++;
      }
    }
  }

  LOG(INFO) << "read " << line_count << " lines, resulting in " << entries
       << " entries" << endl;

  CHECK_GT(entries, 0) << "did not read ANY entry for reference mapping";
  CHECK_GT(line_count, 0);
}

bool ReferenceMapping::isSpecified(word_id_t f) const {
  return misc::Contains(possibleEs_, f);
}

bool ReferenceMapping::isCorrect(word_id_t e, word_id_t f) const {
  auto it = possibleEs_.find(f);
  if (it != possibleEs_.end()) {
    return it->second.find(e) != it->second.end();
  } else {
    // not specified == correct
    return true;
  }
}

size_t ReferenceMapping::correctCount(const Mapping& mapping) const {
  size_t correct = 0;
  for (word_id_t f = 0; f < mapping.getFSize(); ++f) {
    if (mapping.hasF(f) && isCorrect(mapping.getE(f), f)) {
      correct++;
    }
  }
  return correct;
}
