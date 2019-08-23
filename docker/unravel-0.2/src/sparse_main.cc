#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(e_corpus, "", "e_corpus");
DEFINE_string(f_corpus, "", "f_corpus");
DEFINE_string(f_viterbi, "", "f_viterbi");
DEFINE_string(counts, "", "lexicon to use for scoring");
DEFINE_string(out, "", "file to write new lexicon to");
DEFINE_int32(context_pledge, 50, "max context pledge");
DEFINE_int32(counts_pledge, 0, "max lex pledge");
DEFINE_double(base_boost, 0, "boost entries with this factor");
DEFINE_double(boost, 5, "boost entries with this factor");
DEFINE_double(floor, 0.01, "floor for counts");
DEFINE_double(rank_weight, 0.01, "floor for counts");
DEFINE_double(lambda, 1.0, "interpolation");

#include <fst/fstlib.h>
#include <fst/vector-fst.h>
#include <fst/shortest-path.h>
#include <fst/symbol-table.h>
#include <fst/fst.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/algorithm/string.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <limits>
#include <algorithm>

#include "vocab.hh"
#include "lexicon.hh"
#include "lexicon_sparse.hh"
#include "mapping.hh"
#include "kenlm_with_vocab.hh"
#include "misc.hh"
#include "fst_permute.hh"
#include "fst_beamsearch.hh"

size_t max_sentences;
size_t min_observations;

Vocab eVocab, fVocab;

using std::cerr;
using std::endl;

struct Count {
  word_id_t word = 0;
  double count = 0.0;
};

bool CompareCounts(const Count& lhs, const Count& rhs) {
  return lhs.count > rhs.count;
};

struct SimilarityEntry {
  word_id_t e, f;
  double count;
  double context_similarity;
  double count_similarity;
};

bool CompareContexts(const SimilarityEntry& lhs, const SimilarityEntry& rhs) {
  return lhs.context_similarity < rhs.context_similarity;
};

bool CompareLex(const SimilarityEntry& lhs, const SimilarityEntry& rhs) {
  return lhs.count_similarity > rhs.count_similarity;
};

double dist(const std::vector<double> &a, const std::vector<double> &b) {
  CHECK_EQ(a.size(), b.size());
  double result = 0.0;
  for (size_t i=0;i<a.size();++i) {
    if (a[i] == 0 && b[i] == 0) continue;
    result += (a[i]-b[i])*(a[i]-b[i]);
  }
  return result;
}

std::vector<double> getRanks(Vocab * vocab, std::string fn) {
  CHECK_NOTNULL(vocab);
  LOG(INFO) << "reading corpus '" + fn + "' for vocab";
  const auto& sents = vocab->readCorpusAdd(fn);
  std::vector<Count> corpus_counts(vocab->size());
  for (const auto& sent : sents) for (word_id_t w : sent) { corpus_counts[w].word = w; ++corpus_counts[w].count; }
  std::sort(corpus_counts.begin(), corpus_counts.end(), CompareCounts);
  std::vector<double> ranks(vocab->size());
  for (size_t i=0;i<corpus_counts.size();++i) {
    ranks[corpus_counts[i].word] = static_cast<double>(i)/static_cast<double>(vocab->size());
  }
  return ranks;
}

void makeTfIdf(std::vector<std::vector<double>> *cooc_ptr) {
  CHECK_NOTNULL(cooc_ptr);

  auto &cooc = *cooc_ptr;

  std::vector<double> df(cooc[0].size(),0.0);

  for (size_t i=0; i<cooc.size(); ++i) {
    for (size_t j=0; j<cooc[i].size(); ++j) {
      df[j] += cooc[i][j];
    }
  }
  for (size_t i=0; i<cooc.size(); ++i) {
    for (size_t j=0; j<cooc[i].size(); ++j) {
      cooc[i][j] *= ::log(df[j]);
    }
  }
}

std::vector<std::vector<double>> getContextVectors(Vocab *vocab, std::string fn) {
  CHECK_NOTNULL(vocab);
  LOG(INFO) << "reading corpus '" + fn + "' for vocab";
  const auto& sents = vocab->readCorpusAdd(fn);
  vocab->cacheTypes();
  std::vector<std::vector<double>> vectors(vocab->size(), std::vector<double>(vocab->size(), 0.0));
  for (const auto& sent : sents) {
    for (size_t i=0;i<sent.size();++i) {
      for (size_t j=0;j<sent.size();++j) {
        if (i==j) continue;
        // do not use null and unk
        if (vocab->has_null() && sent[j] == vocab->null()) continue;
        if (sent[j] == vocab->unk()) continue;
        ++vectors[sent[i]][sent[j]];
      }
    }
  }
  CHECK_EQ(vectors.size(), vocab->size());
  CHECK_EQ(vectors[0].size(), vocab->size());
  return vectors;
}

// expecting all f and e to be known in vocab
std::vector<std::vector<double>> getContextVectorsFromViterbi(const Vocab &eVocab, const Vocab &fVocab, std::string fn) {
  LOG(INFO) << "reading viterbi paths from '" + fn + "' for vocab";

  misc::IFileStream ifs(fn);

  // cooc[e][f]
  std::vector<std::vector<double>> cooc(fVocab.size(), std::vector<double>(eVocab.size(), 0.0));

  for (std::string str; std::getline(ifs.get(), str);) {
    boost::algorithm::trim(str);
    std::vector<std::string> fields;
    boost::split( fields, str, boost::is_any_of("\t |"), boost::token_compress_on );

    CHECK_EQ(fields.size() % 2, 0);
    CHECK_GE(fields.size(), 8);

    // assume format NUMBER <s>|<s> ...|... ...|... </s>|</s> NUMBER NUMBER NUMBER
    //
    // skip sentence begin
    size_t start_field = 3;
    size_t end_field = fields.size() - 5; // 3 score fields, 1 pair of </s>|</s>

    std::vector<word_id_t> fs, es;

    // read sentence
    for (size_t i=start_field;i<end_field;i+=2) {
      const auto& f_string = fields[i];
      const auto& e_string = fields[i+1];
      word_id_t f = fVocab.getIdFail(f_string);
      word_id_t e = eVocab.getIdFail(e_string);

      fs.push_back(f);
      es.push_back(e);
    }

    CHECK_GT(fs.size(), 0);

    // PPL IF NEEDED
    double ppl = std::stod(fields[fields.size() - 3]) / fs.size();
    CHECK_GT(ppl, 0.0);

    double confidence = 1.0;

    CHECK_EQ(es.size(), fs.size());

    // add coocurences
    for (size_t i=0; i<es.size();++i) {
      for (size_t j=0; j<es.size();++j) {
        if (i==j) continue;

        // do not use null and unk
        if (eVocab.has_null() && es[j] == eVocab.null()) continue;
        if (es[j] == eVocab.unk()) continue;

        CHECK_LT(fs[i], cooc.size());
        CHECK_LT(es[i], cooc[fs[i]].size());
        cooc[fs[i]][es[j]] += confidence;
      }
    }
  }

  CHECK_EQ(cooc.size(), fVocab.size());
  CHECK_EQ(cooc[0].size(), eVocab.size());
  return cooc;
}

void normalizeVectors(vector<vector<double>> *vecs_ptr) {
  auto &vecs = *vecs_ptr;

  for (size_t i=0;i<vecs.size();++i) {
    double sum = 0.0;
    for (double d : vecs[i]) {
      sum += d*d;
    }
    if (sum > 0.0) {
      for ( size_t j=0;j<vecs[i].size();++j) {
        vecs[i][j] /= sum;
      }
    }
  }
}

void addEntries(const std::vector<SimilarityEntry> &entries, size_t e_limit, size_t f_limit, double weight, Lexicon* lex, vector<size_t> *e_count_ptr, vector<size_t> *f_count_ptr) {
  CHECK_NOTNULL(lex);
  CHECK_NOTNULL(e_count_ptr);
  CHECK_NOTNULL(f_count_ptr);

  vector<size_t> &e_count = *e_count_ptr;
  vector<size_t> &f_count = *f_count_ptr;


  for (const auto& entry : entries) {
    //double count = entry.count;
    CHECK_LT(entry.e, eVocab.size());
    CHECK_LT(entry.f, fVocab.size());

    if (!eVocab.isNormalOrNull(entry.e)) continue;
    if (!fVocab.isNormalOrNull(entry.f)) continue;

    if (e_count[entry.e] < e_limit && f_count[entry.f] < f_limit) {
      ++e_count[entry.e];
      ++f_count[entry.f];
      //count = count * FLAGS_boost + FLAGS_base_boost;
      lex->add(entry.e, entry.f, weight);
    }
  }

}


int main(int argc, char** argv) {

  INIT_MAIN("find closest sentences using a lexicon in two corpora\n");

  //////////////////////////////////////////////////
  //auto e_ranks = getRanks(&eVocab, FLAGS_e_corpus);
  LOG(INFO) << "reading context vectors from e_corpus";
  auto e_vectors = getContextVectors(&eVocab, FLAGS_e_corpus);
  LOG(INFO) << "reading corpus to get f_vocab";
  getContextVectors(&fVocab, FLAGS_f_corpus);
  //auto f_ranks = getRanks(&fVocab, FLAGS_f_corpus);
  LOG(INFO) << "reading context vectors from f_viterbi";
  auto f_vectors = getContextVectorsFromViterbi(eVocab, fVocab, FLAGS_f_viterbi);
  //////////////////////////////////////////////////
  CHECK_GT(e_vectors.size(), 0);
  CHECK_GT(f_vectors.size(), 0);

  LOG(INFO) << "normalize vectors";
  //makeTfIdf(&e_vectors);
  //makeTfIdf(&f_vectors);
  normalizeVectors(&e_vectors);
  normalizeVectors(&f_vectors);

  LOG(INFO) << "eVocab.size = " << eVocab.size()
            << " / fVocab.size = " << fVocab.size();
  //////////////////////////////////////////////////
  LexiconSparse lex(&eVocab, &fVocab, 0, FLAGS_lambda);
  lex.init();
  lex.read(FLAGS_counts, true);
  //////////////////////////////////////////////////

  // we might end up with bigger vocabs here
  // only loop over the existing vectors

  LOG(INFO) << "getting marginals";
  vector<double> e_marginal(eVocab.size());
  vector<double> f_marginal(fVocab.size());
  double total_marginal = 0.0;

  LOG(INFO) << "eVocab.size = " << eVocab.size()
            << " / fVocab.size = " << fVocab.size();

  for (size_t e=0;e<e_vectors.size();++e) {
    for (size_t f=0;f<f_vectors.size();++f) {
      double count = lex.lin_score_fast(f,e);
      e_marginal[e] += count;
      f_marginal[f] += count;
      total_marginal += count;
    }
  }

  LOG(INFO) << "reserving enough space for entries vector";
  std::vector<SimilarityEntry> entries;
  entries.reserve(eVocab.size()*fVocab.size());

  LOG(INFO) << "making pairs";
#pragma omp parallel for
  for (size_t e=0;e<e_vectors.size();++e) {
    std::vector<SimilarityEntry> cur_entries(fVocab.size());
    for (size_t f=0;f<f_vectors.size();++f) {
      cur_entries[f].e = e;
      cur_entries[f].f = f;
      double cur_count = lex.lin_score_fast(f,e) + FLAGS_floor;
      double e_other = e_marginal[e] - cur_count;
      double f_other = f_marginal[f] - cur_count;

      cur_entries[f].count_similarity = cur_count / (f_other + e_other);
      cur_entries[f].context_similarity = dist(e_vectors[e], f_vectors[f]);

//      modified_count -= FLAGS_rank_weight * std::abs(e_ranks[e] - f_ranks[f]);
      //modified_count += 0.0001 * e_corpus_counts[e] * f_corpus_counts[f];
//      cur_entry.count = cur_count;
//      cur_entry.modified_count = modified_count;

    }

#pragma omp critical
    {
      entries.insert(entries.end(), cur_entries.begin(), cur_entries.end());
      LOG_EVERY_N(INFO, 100) << "cur e=" << e << " having " << 100.0 * static_cast<double>(entries.size())/static_cast<double>(fVocab.size()*eVocab.size()) << "% of entries done";
    }


  }

  LOG(INFO) << "creating new lex";
  lex.init();


  LOG(INFO) << "sorting " << entries.size() << " entries for lex counts";
  std::sort(entries.begin(), entries.end(), CompareLex);

  {
    vector<size_t> e_count(eVocab.size());
    vector<size_t> f_count(fVocab.size());
    addEntries(entries, FLAGS_counts_pledge, FLAGS_counts_pledge, 1.0, &lex, &e_count, &f_count);
  }

  LOG(INFO) << "sorting " << entries.size() << " entries for context counts";
  std::sort(entries.begin(), entries.end(), CompareContexts);
  {
    vector<size_t> e_count(eVocab.size());
    vector<size_t> f_count(fVocab.size());
    addEntries(entries, FLAGS_context_pledge, FLAGS_context_pledge, 1.0, &lex, &e_count, &f_count);
  }


  LOG(INFO) << "learn";
  lex.learn();
  lex.write(FLAGS_out);
  LOG(INFO) << "done";

  return 0;
}
