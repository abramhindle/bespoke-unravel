#include "embedding.hh"
#include "global.hh"
#include "ngram.hh"
#include "misc.hh"
#include <stdio.h>

#include <limits>
#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/progress.hpp>

Embedding::Embedding() : vocab_(0) {}

const embedding_t& Embedding::getVec(word_id_t word_id) const {
  auto find = embeddings_.find(word_id);
  CHECK(find != embeddings_.end());
  return find->second;
}

bool Embedding::hasVec(word_id_t word_id) const {
  return misc::Contains(embeddings_, word_id);
}

bool pair_comp(pair<word_id_t, double> a, pair<word_id_t, double> b)
{
  return (a.second < b.second);
}

double Embedding::getDis(word_id_t word_a, vector<pair<word_id_t, double> > path)
{
    vector<pair<word_id_t, double> >::iterator it;
    double result = path.back().second;
    for (it = path.begin(); it != path.end(); ++it)
        result += Embedding::diff(getVec(it->first), getVec(word_a));
    return result;
}
        
vector<pair<word_id_t,double> > Embedding::nodesAtNextLevel(const vector<pair<word_id_t, double> > &path, int width)
{
    vector<pair<word_id_t, double> > result;
    vector<pair<word_id_t, double> >::const_iterator it, find_it, max_it;
    for (const auto& word_a : embeddings_)
    {
        find_it = path.end();
        for (it = path.begin(); it != path.end(); it++)
            if (it->first == word_a.first)
                find_it = it;
        if (find_it == path.end()) // a not in path
        {
            double score = getDis(word_a.first, path);
            result.push_back(make_pair(word_a.first, score));
        }
    }
    partial_sort(result.begin(), result.begin() + width, result.end(), pair_comp);
    return result;
}

bool path_comp(const vector<pair<word_id_t, double> > &path_a, const vector<pair<word_id_t, double> > &path_b)
{
    vector<pair<word_id_t, double> >::const_iterator it_a = path_a.end(), it_b = path_b.end();
    while (it_a != path_a.begin() && it_b != path_b.begin())
    {
        it_a--; it_b--;
        if (it_a->second != it_b->second)
            return (it_a->second < it_b->second);
    }
    return false;
}
vector<vector<pair<word_id_t,double> > > Embedding::beamsearch(int width)
{
    vector<vector<pair<word_id_t,double> > > paths;
    vector<vector<pair<word_id_t,double> > >::iterator path, start_it, end_it;
    vector<pair<word_id_t, double> >::iterator it;
    for (const auto& word_a : embeddings_)
    //if (word_a.first < 100)
    {
        vector<pair<word_id_t, double> > root_path;
        root_path.push_back(make_pair(word_a.first, 0.0));
        paths.push_back(root_path);
    }
    int DEPTH = embeddings_.size()-1;
    for (int depth = 0; depth < DEPTH; ++ depth) 
    {
        size_t i = 0;
        vector<vector<pair<word_id_t, double> > > new_paths;
        for (path = paths.begin(); path != paths.end(); ++path)
        {
            VLOG(1) << "working on " << i++ << "/" << paths.size();

            vector<pair<word_id_t, double> > all_paths = nodesAtNextLevel(*path, width);
            int cnt = 0;
            for (it = all_paths.begin(); it != all_paths.end() && (cnt++) < width; ++it)
            {
                vector<pair<word_id_t, double> > new_path(*path);
                new_path.push_back(*it);
                new_paths.push_back(new_path);
            }
        }
        partial_sort(new_paths.begin(), new_paths.begin() + width, new_paths.end(), path_comp);
        new_paths.resize(width);
        paths = new_paths;
        LOG(INFO) << "depth " << depth << ":";
        for (it = paths.begin()->begin(); it != paths.begin()->end(); ++it)
            LOG(INFO) << " " << getVocab().getWord(it->first);
    }
    /*
    LOG(INFO) << "paths for '" << getVocab().getWord(word_a.first) << "':" << std::endl;
    for (path = paths.begin(); path != paths.end(); ++path)
    {
        LOG(INFO) << "  PATH" << std::endl;
        for (it = path->begin(); it != path->end(); it++)
            LOG(INFO) << "    '" << getVocab().getWord(it->first) << "' with diff:" << (it->second) << endl;
    }
    */
    return paths;
}

void Embedding::init_closest(int K)
{
  if (K==0) {
    LOG(INFO) << "not setting up closest words, since k=" << K;
    return;
  }

  // to keep best K elements in closest vector O(n*K)
  // could be improved by using a heap to O(n*log(K))
  LOG(INFO) << "setting up closest words lookup" << std::endl;
  // store distance of closest word 
  for (const auto& word_a : embeddings_) {
    word_id_t word_id_a = word_a.first;
    for (const auto& word_b : embeddings_) {
      word_id_t word_id_b = word_b.first;
      if (word_id_a == word_id_b) continue;
      double score = Embedding::diff(getVec(word_id_a), getVec(word_id_b));
      if (closest_[word_id_a].size() < K)
        closest_[word_id_a].push_back(std::make_pair(word_id_b, score));
      else
      {
        vector<pair<word_id_t, double> >::iterator it, max_it;
        max_it = closest_[word_id_a].begin();
        for (it = closest_[word_id_a].begin(); it != closest_[word_id_a].end(); ++it)
            if (it->second > max_it->second)
                max_it = it;
        if (score < max_it->second)
        {
            max_it->first = word_id_b;
            max_it->second = score;
        }
      }
    }
  sort(closest_[word_id_a].begin(), closest_[word_id_a].end(), pair_comp);
  VLOG(1) << "Closest K words for " << word_id_a << " : ";
  vector<pair<word_id_t, double> >::iterator it;
  for (it = closest_[word_id_a].begin(); it != closest_[word_id_a].end(); ++it)
      VLOG(1) << "    id=" << it->first  << " with score: " << it->second << std::endl;
  }
  LOG(INFO) << "setting up closest words lookup done" << std::endl;
}

void Embedding::read(std::string fn, Vocab& vocab,
                bool verbose, bool add_new, int K) {
  vocab_ = &vocab;
  LOG(INFO) << "reading embeddings from '" << fn << "'";

  CHECK_GT(vocab_->size(), 0);

  FILE *f;
  f = fopen(fn.c_str(), "rb");
  if (f == NULL) {
    LOG(FATAL) << "input file '" << fn << "' not found";
  }

  // file contains size info as header
  long long words;
  long long vec_dim;
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &vec_dim);

  LOG(INFO) << "reading vectors from '" << fn << "', having " << words << " words with dimension " << vec_dim;

  size_t skipped_words = 0;

  for (size_t b = 0; b < words; b++) {
    size_t a = 0;
    char cur_word[maximum_word_length];
    while (1) {
      cur_word[a] = fgetc(f);
      // read until space
      if (feof(f) || (cur_word[a] == ' ')) break;
      if (a > maximum_word_length || cur_word[a] != '\n') a++;
    }
    cur_word[a] = '\0';

    // read actual vector
    std::vector<float> cur_vec(vec_dim, 0.0);
    fread(&cur_vec[0], sizeof(float), vec_dim, f);
    normalize(&cur_vec);

    // make sure that it is really normalzied
    CHECK_NEAR(dot(cur_vec, cur_vec), 1.0, 0.0001);
    VLOG_EVERY_N(1,10000) << "reading line " << b << " word '" << cur_word << "'" << std::endl;

    // store
    if (add_new) {
      word_id_t word_id = vocab_->addWord(cur_word);
      embeddings_[word_id] = cur_vec;
    } else {
      if (vocab_->containsWord(cur_word)) {
        word_id_t word_id = vocab_->getId(cur_word);
        embeddings_[word_id] = cur_vec;
      } else {
        ++skipped_words;
      }
    }

  }
  fclose(f);
  LOG(INFO) << "reading done. added " << embeddings_.size() << " entries, skipped " << skipped_words << " entries" << std::endl;
  init_closest(K);
}

Embedding::~Embedding() {}

Vocab& Embedding::getVocab() { return *vocab_; }

const Vocab& Embedding::getVocabConst() const { return *vocab_; }
