#ifndef GLOBAL_OPTIONS_HH
#define GLOBAL_OPTIONS_HH

#include <lm/model.hh>

typedef unsigned char word_type_t;
typedef unsigned int word_id_t;
typedef size_t class_id_t;
typedef unsigned char order_t;
typedef unsigned int count_t;
typedef float score_t;

#define BS_MAX_ORDER 10
const word_id_t MAPPING_NOT_MAPPED = std::numeric_limits<word_id_t>::max(); //255; //(1024UL*1024UL*1024UL)

// mainly used in hypothesis_struct.hh
// should fit number of words
const size_t MAX_LEN = 10000UL;
const size_t INVALID_PARENT = (1024UL*1024UL*1024UL);

inline void check_global_options_integrity() {
//    CHECK_EQ(KENLM_MAX_ORDER, BS_MAX_ORDER);
}

#endif // GLOBAL_OPTIONS_HH
