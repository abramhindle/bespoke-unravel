#ifndef HYPOTHESISSTRUCT_H_
#define HYPOTHESISSTRUCT_H_

#include "global.hh"

// select suitable types

// id for hyps (should fit histogram size)
typedef unsigned int hyp_id_t;

struct map_pair {
    word_id_t e, f;
};

struct partial_hyp_body {
    partial_hyp_body(size_t number_heu) : cor_count(0) {
        scores.resize(number_heu);
    }
    hyp_id_t parent_id;
    word_id_t e_word_count[MAX_LEN];
    map_pair word_pairs[MAX_LEN];
    std::vector<score_t> scores;
    score_t score;
    word_id_t cor_count;
};

struct partial_hyp_head {
    partial_hyp_head(size_t number_heu) : cor_count(0) {
        scores.resize(number_heu);
    }
    hyp_id_t parent_id;
    word_id_t e;
    word_id_t f;
    std::vector<score_t> scores;
    score_t score;
    word_id_t cor_count;
};

inline bool cmp_partial_hyp_head(const partial_hyp_head & lhs, const partial_hyp_head & rhs) {
    if (rhs.parent_id==INVALID_PARENT) return (lhs.parent_id!=INVALID_PARENT);
    if (lhs.parent_id==INVALID_PARENT) return false;
    return lhs.score > rhs.score;
};

inline bool cmp_partial_hyp_body(const partial_hyp_body & lhs, const partial_hyp_body & rhs) {
    return lhs.score > rhs.score;
};

#endif /* HYPOTHESISSTRUCT_H_ */
