#include "misc.hh"
#include <glog/logging.h>

#include <sstream>
#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "kenlm_with_vocab.hh"

#ifdef __APPLE__
#include <mach/clock.h>
#include <mach/mach.h>
struct task_basic_info t_info;
#endif
using std::cerr;
using std::endl;

namespace misc {

bool any_of(word_id_t w, const std::vector<word_id_t>& a) {
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] == w) return true;
  return false;
}

double calcNumberPossibilitiesLog(int n, int k) {
  double result = 0;
  int x = n;
  while (x > n - k) {
    result += log10(x);
    --x;
  }
  return result;
}
//////////////////////////////////////////////////////////////////////////////
//
// process_mem_usage(double &, double &) - takes two doubles by reference,
// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
//
// On failure, returns 0.0, 0.0

void process_mem_usage(double& vm_usage, double& resident_set) {
  using std::ios_base;
  using std::ifstream;
  using std::string;

  vm_usage = 0.0;
  resident_set = 0.0;

  // 'file' stat seems to give the most reliable results
  ifstream stat_stream("/proc/self/stat", ios_base::in);

  // dummy vars for leading entries in stat that we don't care about
  //
  string pid, comm, state, ppid, pgrp, session, tty_nr;
  string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  string utime, stime, cutime, cstime, priority, nice;
  string O, itrealvalue, starttime;

  // the two fields we want
  //
  unsigned long vsize;
  long rss;

  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
      tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
      stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >>
      starttime >> vsize >> rss;  // don't care about the rest

  stat_stream.close();

  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024;  // in case x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;
}

double mem_usage() {
#ifdef __linux__
  double vm_usage, resident_set;
  process_mem_usage(vm_usage, resident_set);
  return 1024 * vm_usage;
#elif __APPLE__
  mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

  if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO,
                                (task_info_t) & t_info, &t_info_count)) {
    return -1.0;
  }

  return double(t_info.resident_size);

#endif
}

std::vector<word_id_t> sorted_normal_words(const KenLmWithVocab& lm) {
  std::vector<word_id_t> sorted_words;
  std::vector<WordScore> wsps;
  for (word_id_t w : lm.getVocabConst().getNormalWords()) {
    wsps.push_back(WordScore(w, lm.getScore(lm.getNullContextState(), w)));
  }
  std::sort(wsps.begin(), wsps.end(), compareWordScorePairBigger);
  for (WordScore ws : wsps) {
    sorted_words.push_back(ws.first);
    LOG(INFO) << lm.getVocabConst().getWord(ws.first);
  }
  return sorted_words;
}

std::vector<word_id_t> sorted_normal_words(
    const Vocab& vocab, const std::vector<std::vector<word_id_t> >& snts) {
  std::vector<word_id_t> sorted_words;
  std::vector<score_t> p = p_unigram(vocab, snts);
  std::vector<WordScore> wsps;
  for (word_id_t w : vocab.getNormalWords()) {
    wsps.push_back(WordScore(w, p[w]));
  }
  std::sort(wsps.begin(), wsps.end(), compareWordScorePairBigger);
  for (WordScore ws : wsps) {
    sorted_words.push_back(ws.first);
    LOG(INFO) << vocab.getWord(ws.first);
  }
  return sorted_words;
}

std::vector<score_t> p_unigram(const KenLmWithVocab& lm) {
  std::vector<score_t> p(lm.getVocabConst().size(), 0.0);
  for (word_id_t w = 0; w < lm.getVocabConst().size(); ++w) {
    p[w] = lm.getScore(lm.getNullContextState(), w);
  }
  return p;
}

std::vector<score_t> p_unigram(
    const Vocab& vocab, const std::vector<std::vector<word_id_t> >& snts) {
  std::vector<score_t> p(vocab.size(), 0.0);
  score_t N = 0;
  for (const std::vector<word_id_t>& snt : snts) {
    for (word_id_t w : snt) {
      ++p[w];
      ++N;
    }
  }
  for (word_id_t w = 0; w < vocab.size(); ++w) {
    p[w] /= N;
  }
  return p;
}

} /* namespace Misc */
