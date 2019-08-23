#ifndef EMBEDDING_HH_
#define EMBEDDING_HH_

#include <boost/iostreams/filtering_stream.hpp>
#include <set>

#include "vocab.hh"
#include "misc.hh"
#include "extension_order.hh"

using std::vector;
using std::pair;

typedef std::vector<float> embedding_t;
using std::vector;
using std::pair;
using std::make_pair;

class ExtensionOrder;

class Embedding {

  const size_t maximum_word_length = 1024;

 public:
  Embedding();
  void read(std::string fn, Vocab& vocab,
            bool verbose, bool add_new, int K);
  const embedding_t& getVec(word_id_t word_id) const;
  bool hasVec(word_id_t word_id) const;
  virtual ~Embedding();
  Vocab& getVocab();
  const Vocab& getVocabConst() const;
  vector<vector<pair<word_id_t,double> > > beamsearch(int width);

  std::vector<pair<word_id_t, double>> getClosest(word_id_t word_id) const {
    std::map<word_id_t,std::vector<pair<word_id_t,double>>>::const_iterator it = closest_.find(word_id);
    CHECK(it != closest_.end());
    return it->second;
  };

  static inline void logVec(const embedding_t& a) {
    LOG(INFO) << "vector =====" << std::endl;
    for (size_t i=0;i<a.size();++i) {
      LOG(INFO) << a[i] << std::endl;
    }
    LOG(INFO) << "============" << std::endl;
  };

static inline float dot(const embedding_t& a, const embedding_t& b) {
  CHECK_EQ(a.size(), b.size());
	float result = 0.0;
	for (size_t i=0;i<a.size();++i) result += a[i]*b[i];
	return result;
}

static inline void scale(double val, embedding_t* a) {
  CHECK_NOTNULL(a);
	for (size_t i=0;i<a->size();++i) (*a)[i] = val * (*a)[i];
}

static inline void add(const embedding_t& a, embedding_t *b) {
  CHECK_NOTNULL(b);
  CHECK_EQ(a.size(), b->size());
	for (size_t i=0;i<a.size();++i) (*b)[i] = (*b)[i] + a[i];
}

static inline void diff(const embedding_t& a, const embedding_t& b, embedding_t *c) {
  CHECK_NOTNULL(c);
  CHECK_EQ(a.size(), b.size());
  CHECK_EQ(a.size(), c->size());
	for (size_t i=0;i<a.size();++i) (*c)[i] = (a[i]-b[i]);
}

static inline float diff(const embedding_t& a, const embedding_t& b) {
  CHECK_EQ(a.size(), b.size());
  double result = 0.0;
	for (size_t i=0;i<a.size();++i) result += (a[i]-b[i])*(a[i]-b[i]);
  return sqrt(result);
}

static void normalize(std::vector<float>* a) {
	float len = dot(*a,*a);
	if (len == 0) return;
	for (size_t i=0;i<a->size();++i) (*a)[i] /= sqrt(len);
}

 private:
  Vocab* vocab_;
  std::map<word_id_t, embedding_t > embeddings_;
  std::map<word_id_t, vector<pair<word_id_t, double> > > closest_;
  void init_closest(int K);
  vector<pair<word_id_t, double> > nodesAtNextLevel(const vector<pair<word_id_t, double> > &path, int width);
  double getDis(word_id_t word_a, vector<pair<word_id_t, double> > path);
};

#endif /* EMBEDDING_HH_ */
