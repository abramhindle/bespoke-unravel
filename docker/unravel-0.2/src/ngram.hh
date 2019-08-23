#ifndef NGRAM_H_
#define NGRAM_H_

#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "global.hh"

struct Ngram {
    order_t order_;
    count_t count;
    word_id_t t[BS_MAX_ORDER];

    Ngram(order_t order) {
        order_ = order;
    }

    virtual ~Ngram() {
    }

    bool operator<(const Ngram& A) const {
        for (order_t i=0; i<order_-1; ++i) {
            if (t[i] < A.t[i])
                return true;
            else if (t[i] > A.t[i])
                return false;
        }
        return t[order_-1]<A.t[order_-1];
    }

    // convert ngram to size_t with given vocab size
    size_t operator %(const size_t& right) const {
        size_t result = 0;
        size_t factor = 1;
        for (order_t i=0;i<order_;++i) {
            result += t[i]*factor;
            factor*=right;
        }
        return result;
    }

};

inline bool ngramCountBiggerRelation(const Ngram &ngram1, const Ngram &ngram2) {
    return ngram1.count > ngram2.count;
}

#endif
