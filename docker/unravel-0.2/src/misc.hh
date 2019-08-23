#ifndef MISC_H_
#define MISC_H_

#include "misc_io.hh"
#include "misc_time.hh"
#include "misc_prettyprint.hh"
#include "statistics.hh"

#include <glog/logging.h>
#include "config_options.hh"
#include "hypothesis_struct.hh"
#include "ngram.hh"

#include <iomanip>
#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <fstream>
#include <ctime>
#include <time.h>


class Vocab;
class KenLmWithVocab;


#define TWO_STREAMS(S1, S2, IN) S1 << IN ; \
  S2 << IN ;

// ANSI escape codes.
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define NORMAL   "\x1b[0m"

#define MISC_VARINFO(X) VLOG(1) << #X << " has " << sizeof(X) << " bytes" << std::endl;
#define MISC_ENVINFO(X) VLOG(1) << #X << " has value " << X << std::endl;

// not sure what this should do
// but produced warning: unused variable 'local' [-Wunused-variable]
// time_t t; t = time(NULL);
//  struct tm *local = localtime(&t);
#define INIT_MAIN(DESCRIPTION) { \
  google::SetUsageMessage(DESCRIPTION); \
  google::ParseCommandLineFlags(&argc, &argv, true); \
  google::InitGoogleLogging(argv[0]); \
  time_t time_run = time(NULL); \
  time_t time_compile = misc::compile_time(); \
  double age = difftime(time_run, time_compile); \
  \
  std::string time_compile_str = asctime(localtime(&time_compile)); \
  time_compile_str.resize(time_compile_str.size() - 1); \
  \
  std::string time_run_str = asctime(localtime(&time_run)); \
  time_run_str.resize(time_run_str.size() - 1); \
  \
  LOG(INFO) << "revision '" << GIT_REVISION << "' compiled at " << time_compile_str << " (age " << age << "s)"; \
  \
  MISC_VARINFO(size_t); \
  MISC_VARINFO(char); \
  MISC_VARINFO(short int); \
  MISC_VARINFO(int); \
  MISC_VARINFO(uint64_t); \
  MISC_VARINFO(long int); \
  MISC_VARINFO(long long int); \
  MISC_VARINFO(bool); \
  MISC_VARINFO(float); \
  MISC_VARINFO(double); \
  MISC_VARINFO(long double); \
  MISC_VARINFO(long long); \
  MISC_VARINFO(score_t); \
  MISC_VARINFO(word_id_t); \
  MISC_VARINFO(order_t); \
  MISC_VARINFO(count_t); \
  MISC_VARINFO(Ngram); \
}

using std::cerr;
using std::endl;

namespace misc {

typedef std::pair<word_id_t, score_t> WordScore;
inline bool compareWordScorePair(const WordScore& i, const WordScore& j) {
  return i.second < j.second;
}

inline bool compareWordScorePairBigger(const WordScore& i,
                                       const WordScore& j) {
  return i.second > j.second;
}


extern word_id_t e_sent, f_sent, e_sentend, f_sentend, e_unk, e_null, f_null;

std::ostream& clear_line();
std::ostream& clear_line(std::ostream& ostr);

std::ostream& log();
std::ostream& log(std::ostream& ostr);
std::ostream& update(double interval, bool force);

std::string intToStrZeroFill(size_t iteration, size_t length);
std::string intToHumanStr(size_t num);

void start();

// containers
// helpers for containers
template <template<class,class,class...> class C, typename K, typename V, typename... Args> V GetDefault(const C<K,V,Args...>& m, K const& key, const V & defval) {
  typename C<K,V,Args...>::const_iterator it = m.find( key );
  if (it == m.end()) return defval;
  return it->second;
}

// helpers for containers
template <template<class,class,class...> class C, typename K, typename V, typename... Args> bool Contains(const C<K,V,Args...>& m, K const& key) {
  typename C<K,V,Args...>::const_iterator it = m.find( key );
  return (it != m.end());
}

template<typename T> inline std::string getBinaryStr(T val) {
  std::string str_str;
  str_str.resize(sizeof(T));
  char* start = (char*) &val;
  for (size_t i=0;i<sizeof(T);++i) {
    str_str[i] = *start;
    start++;
  }
  return str_str;
}

template<typename T> inline std::string ContainerToString(T container) {
  std::ostringstream oss;
  bool first = true;

  oss << "{";
  for (const auto v : container) {
    if (first) {
      first = false;
    } else {
      oss << ", ";
    }
    oss << std::to_string(v);
  }
  oss << "}";
  return oss.str();
}

// symbols
bool any_of(word_id_t w, const std::vector<word_id_t>& a);
bool specialSymbolOK(word_id_t f, word_id_t e);

// stuff
bool compareWordScorePair(const WordScore& i, const WordScore& j);
double calcNumberPossibilitiesLog(int n, int k);
void fillSetWithIntegers(std::set<size_t>& set, size_t maxValue);
inline bool compare_pair_second ( const std::pair<size_t,size_t>& l, const std::pair<size_t,size_t>& r) { return l.second > r.second; }
void process_mem_usage(double& vm_usage, double& resident_set);
double mem_usage();

// does not support both values to be -inf, this is not a bug, it's a feature
inline score_t add_log_scores(score_t a, score_t b) {
  CHECK(std::isfinite(a) || std::isfinite(b));
  if (a > b) {
    return a+log1p(exp(b-a));
  } else {
    return b+log1p(exp(a-b));
  }
}

// this supports both values to be -inf
inline score_t add_log_scores_0(score_t a, score_t b) {
  if (((- std::numeric_limits<score_t>::infinity()) == a) && ((- std::numeric_limits<score_t>::infinity()) == b)) {
    return - std::numeric_limits<score_t>::infinity();
  }
  if (a > b) {
    return a+log1p(exp(b-a));
  } else {
    return b+log1p(exp(a-b));
  }
}


std::vector<word_id_t> sorted_normal_words(const KenLmWithVocab& lm);
std::vector<word_id_t> sorted_normal_words(
    const Vocab& vocab, const std::vector<std::vector<word_id_t> >& snts);
std::vector<score_t> p_unigram(const KenLmWithVocab& lm);
std::vector<score_t> p_unigram(
    const Vocab& vocab, const std::vector<std::vector<word_id_t> >& snts);

} /* namespace Misc */
#endif /* MISC_H_ */
