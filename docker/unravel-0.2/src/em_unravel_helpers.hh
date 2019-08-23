Vocab eVocab_, fVocab_;
// 0 = read from
// 1 = first accumulation lexicon
LexiconSparse *lex_;
std::vector<LexiconSparse *> accumulation_lex_;

KenLmWithVocab *klm_;
std::mutex mtx_;

// corpus
std::vector<std::string> e_sentences_, f_sentences_;
std::vector<std::vector<word_id_t>> e_sentences_idx_, f_sentences_idx_;

const size_t maxSentenceLength_ = 128;

typedef std::bitset<maxSentenceLength_> CoverageVector;

struct statistics {
  size_t train_tokens_total = 0, train_tokens_correct = 0;
  size_t test_tokens_total = 0, test_tokens_correct = 0;
  size_t insertions_total = 0, deletions_total = 0;
  double bleu_total = 0.0;
  std::atomic<size_t> cur_sent;
  std::atomic<size_t> sentences_done;
  statistics()
      : train_tokens_total(0),
        train_tokens_correct(0),
        test_tokens_total(0),
        test_tokens_correct(0),
        insertions_total(0),
        deletions_total(0),
        cur_sent(0),
        sentences_done(0) {}

  statistics(const statistics &other)
      : train_tokens_total(other.train_tokens_total),
        train_tokens_correct(other.train_tokens_correct),
        test_tokens_total(other.test_tokens_total),
        test_tokens_correct(other.test_tokens_correct),
        insertions_total(0),
        deletions_total(0),
        cur_sent(0),
        sentences_done(0) {}

  void reset() {
    bleu_total = 0.0;
    train_tokens_total = 0;
    train_tokens_correct = 0;
    test_tokens_total = 0;
    test_tokens_correct = 0;
    insertions_total = 0;
    deletions_total = 0;
    cur_sent = 0;
    sentences_done = 0;
  }
};

template <size_t N> bool CompareCoverage(const std::bitset<N> &lhs, const std::bitset<N> &rhs) {
  for (size_t i = 0; i < N; ++i) {
    if (lhs[i] == rhs[i]) continue;
    return lhs[i] < rhs[i];
  }
  return false;
}

CoverageVector MakeFullCoverageVector(size_t size) {
  CoverageVector full_coverage;
  for (size_t i = 0; i < size; ++i) full_coverage.set(i);
  return full_coverage;
}


struct HypNode {
  word_id_t e, f;  // having f in here is not really consistent
  lm::ngram::State lm_state;
  fst::LogArc::StateId pred_output_fst_state, output_fst_state;

  unsigned short cardinality = 0;
  unsigned short num_insertions = 0;
  unsigned short num_deletions = 0;

  CoverageVector coverage;
  bool last_insertion = false;
  score_t score, delta;

  bool operator==(const HypNode &x) const {
    return lm_state == x.lm_state && coverage == x.coverage && cardinality == x.cardinality &&
           (e == eVocab_.null()) == (x.e == eVocab_.null()) &&
           (f == fVocab_.null()) == (x.f == fVocab_.null());
  }

  // equal if lm_state is the same, and if deletion state is the same
  bool operator<(const HypNode &x) const {
    if (num_insertions == x.num_insertions) {
      if (num_deletions == x.num_deletions) {
        if (last_insertion == x.last_insertion) {
          if (lm_state == x.lm_state) {
            if ((e == eVocab_.null()) == (x.e == eVocab_.null())) {
              if (cardinality == x.cardinality) {
                return CompareCoverage(coverage, x.coverage);
              } else {
                return cardinality < x.cardinality;
              }
            } else {
              return (e == eVocab_.null()) < (x.e == eVocab_.null());
            }
          } else {
            return lm_state < x.lm_state;
          }
        } else {
          return last_insertion < x.last_insertion;
        }
      } else {
        return num_deletions < x.num_deletions;
      } 
    } else {
      return num_insertions < x.num_insertions;
    }
  }

  HypNode() {}

  HypNode(KenLmWithVocab &lm)
      : 
        e(eVocab_.sos()),
        f(fVocab_.sos()),
        pred_output_fst_state(0),
        output_fst_state(0),
        cardinality(0),
        num_insertions(0),
        num_deletions(0),
        coverage(0),
        last_insertion(false),
        score(0.0),
        delta(0.0) {
    // set <s> context
    lm_state = lm.getModel().NullContextState();
    lm.getModel().Score(lm_state, eVocab_.sos(), lm_state);
  }

  HypNode(word_id_t e, word_id_t f, CoverageVector coverage, size_t cardinality,
          size_t output_fst_state, lm::ngram::State lm_state, score_t score,
          score_t delta, unsigned short num_insertions, unsigned short num_deletions, bool last_insertion)
      : 
        e(e),
        f(f),
        lm_state(lm_state),
        pred_output_fst_state(output_fst_state),
        output_fst_state(0),
        cardinality(cardinality),
        num_insertions(num_insertions),
        num_deletions(num_deletions),
        coverage(coverage),
        last_insertion(last_insertion),
        score(score),
        delta(delta) {}
};

typedef std::map<HypNode, size_t> HypToStateMap;
//typedef std::unordered_map<HypNode, size_t> HypToStateMap;

void AddSpecialCandidates(word_id_t f, std::set<word_id_t> *candidates, size_t max_deletions) {
  CHECK_NOTNULL(candidates);
  if (f == fVocab_.sos()) {
    candidates->insert(eVocab_.sos());
  } else if (f == fVocab_.eos()) {
    candidates->insert(eVocab_.eos());
  } else if (f != fVocab_.unk() && f != fVocab_.null()) {
    if (max_deletions > 0) candidates->insert(eVocab_.null());
  }
}

bool HypNodeCompareScore(const HypNode &lhs, const HypNode &rhs) {
  return lhs.score < rhs.score;
}

std::string HypToString(const HypNode &cur_hyp, const Vocab &eVocab, const Vocab &fVocab, size_t input_sentence_size) {
  std::stringstream ss;
    ss << "[hyp] coverage="
      << cur_hyp.coverage.to_string().substr(cur_hyp.coverage.size()-input_sentence_size)
      << " " << fVocab.getWord(cur_hyp.f) << " -> " << eVocab.getWord(cur_hyp.e)
      << " output_fst_state = " << cur_hyp.output_fst_state
      << " pred_output_fst_state = " << cur_hyp.pred_output_fst_state
      << " insertions=" << cur_hyp.num_insertions
      << " deletions=" << cur_hyp.num_deletions
      << " card=" << cur_hyp.cardinality
      << " score=" << cur_hyp.score;
    return ss.str();
}

// helper methods for jumps
// ////////////////////////////////////////////////////////////////////////////
inline size_t FirstUncoveredPosition(const HypNode &hyp, size_t sentence_len) {
  size_t first_uncovered = 0;
  for (size_t i = 0; i < sentence_len; ++i) {
    if (!hyp.coverage[i]) {
      first_uncovered = i;
      break;
    }
  }
  return first_uncovered;
}

inline size_t JumpSize(const HypNode &hyp, size_t pos, size_t sentence_len) {
  CHECK_EQ(hyp.coverage[pos], 0);

  size_t first_uncovered = FirstUncoveredPosition(hyp, sentence_len);

  if (first_uncovered > pos) {
    return 0;
  } else {
    return pos - first_uncovered;
  }
}

inline bool JumpPossible(const HypNode &hyp, size_t pos, size_t sentence_len,
                  size_t window) {
  // already covered, then not possible
  if (hyp.coverage[pos]) return false;

  size_t first_uncovered = FirstUncoveredPosition(hyp, sentence_len);

  // window doesn't allow this big jump
  if (first_uncovered + window < pos) return false;

  CHECK_LT(pos, sentence_len);

  // the last position is only allowed, if we covered everything else before
  if (static_cast<size_t>(hyp.cardinality + hyp.num_deletions + 1) < sentence_len &&
      static_cast<size_t>(pos + 1) == sentence_len)
    return false;

  return true;
}

// ////////////////////////////////////////////////////////////////////////////

// are these two states recombinable?
inline bool CanRecombine(const HypNode &a, const HypNode &b) {
  if (a.last_insertion != b.last_insertion) return false;
  if (a.num_insertions != b.num_insertions) return false;
  if (!(a.lm_state == b.lm_state)) return false;
  if ((a.e == eVocab_.null()) != (b.e == eVocab_.null()))
    return false;
  if (CompareCoverage(a.coverage, b.coverage)) return false;
  if (CompareCoverage(b.coverage, a.coverage)) return false;
  return true;
}
