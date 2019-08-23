#ifndef CONSTRAINT_H_
#define CONSTRAINT_H_

#include "misc.hh"
#include "classes.hh"
#include "cooc.hh"
#include "mapping.hh"

class Constraint {
public:
  enum ConstraintAcceptCode {
    FORCE_KEEP = 0,
    REJECT = 1,
    DONT_CARE = 2
  };
    virtual ~Constraint() {
    }
    virtual ConstraintAcceptCode accept(size_t e, size_t f, const Mapping & mapping, const partial_hyp_body & body, partial_hyp_head & head) = 0;
};

// need to work on that
class HomophonicConstraint : public Constraint {
    ConstraintAcceptCode accept(size_t e, size_t f, const Mapping & mapping, const partial_hyp_body & body, partial_hyp_head & head) {
        (void) mapping; // unused
        (void) e;
        (void) f;
        (void) body;
        (void) head;
        // need sourceVectorRef ??
        // bool accept = (sourceVectorRef[i].e_word_count[e]<limit_e[e]) && (special_e_words.count(e)==0);
        return DONT_CARE;
    }
};

// need to work on that
class ClassesConstraint : public Constraint {
    public:
        ClassesConstraint(const Classes & eClasses,
            const Classes & fClasses,
            const ExtensionOrder & extOrder,
            /*size_t max_penalty,*/
            bool verbose = false) : 
          eClasses_(eClasses),
          fClasses_(fClasses),
          extOrder_(extOrder),
          /*max_penalty_(max_penalty),*/
          verbose_(verbose) {
            (void) verbose; // unused
            LOG(INFO) << "Setting up ClassesConstraint with fClasses=" << fClasses_.getMaxClass() << std::endl;
            firstMappedWordInClass_.resize(fClasses_.getMaxClass(),0);

            LOG(INFO) << "Looping over extOrder" << std::endl;
            for (size_t i=0;i<extOrder_.size();++i) {
                size_t cur_f = extOrder.getWordAt(i);
                size_t cur_f_class = fClasses_.getClass(cur_f);
                LOG(INFO) << "i=" << i << " f=" << cur_f << " '" << fClasses.getVocabConst().getWord(cur_f) << "' f_class=" << cur_f_class << std::endl;
                CHECK_NE(cur_f_class, CLASS_NOT_FOUND);
                if (firstMappedWordInClass_[cur_f_class]==0) firstMappedWordInClass_[cur_f_class] = cur_f;
            }
        }

        ConstraintAcceptCode accept(size_t e, size_t f, const Mapping & mapping, const partial_hyp_body & body, partial_hyp_head & head) {
            (void) body;
            (void) head;
//            if (verbose_) LOG(INFO) << "ClassConstraint: checking " << eClasses_.getVocabConst().getWord(e) << " vs " << fClasses_.getVocabConst().getWord(f) << std::endl;
            // find first f' with class[f'] == class[f]
            // find e' = phi(f')
            // accept e if class[e] = class[e']
            class_id_t fc = fClasses_.getClass(f);
            if (verbose_) LOG(INFO) << "class_f=" << fc << std::endl;
            word_id_t f_prime = firstMappedWordInClass_[fc];
            if (verbose_) LOG(INFO) << "f_prime=" << f_prime << std::endl;
            word_id_t e_prime = mapping.getE(f_prime);
            if (verbose_) LOG(INFO) << "e_prime=" << e_prime << std::endl;

            if (e_prime != MAPPING_NOT_MAPPED) {
                class_id_t e_prime_c = eClasses_.getClass(e_prime);
                if (verbose_) {
                    LOG(INFO) << "e_prime_c=" << e_prime_c << std::endl;
                    LOG(INFO) << "e_c=" << eClasses_.getClass(e) << std::endl;
                }
                //if (e_prime_c != eClasses_.getClass(e)) {
                //    head.classes_penalty++;
                //}
            } else {
                if (verbose_) LOG(INFO) << "e_prime not yet mapped" << std::endl;
            }
            // not functioning right now
            return REJECT;
//            return head.classes_penalty < max_penalty_;
        }
    private:
        const Classes & eClasses_;
        const Classes & fClasses_;
        const ExtensionOrder & extOrder_;
        std::vector<size_t> firstMappedWordInClass_;
        //const size_t max_penalty_;
        const bool verbose_;
};

class UnigramConstraint : public Constraint {
public:
    UnigramConstraint(const Counts & e, const Counts & f, double rank_diff_max, size_t cnt_diff_max, size_t card_max, bool verbose=false) : e_(e), f_(f), rank_diff_max_(rank_diff_max), cnt_diff_max_(cnt_diff_max), card_max_(card_max) {
        const std::vector<Ngram> & es = e.getNgrams();
        const std::vector<Ngram> & fs = f.getNgrams();

        (void) e_;
        (void) f_;

        // reserve vectors for vocab
        LOG(INFO) << "UnigramConstraint: setting up vectors" << std::endl;
        e_counts_.resize(e.getVocabConst().size());
        f_counts_.resize(f.getVocabConst().size());
        e_ranks_.resize(e.getVocabConst().size());
        f_ranks_.resize(f.getVocabConst().size());

        LOG(INFO) << "UnigramConstraint: filling vectors" << std::endl;

        std::vector<std::pair<size_t,size_t> > e_pairs(e_counts_.size());
        std::vector<std::pair<size_t,size_t> > f_pairs(f_counts_.size());

        // fill vec
        for (size_t i=0;i<es.size();++i) {
            e_counts_[es[i].t[0]] = es[i].count;
            e_pairs[es[i].t[0]] = std::pair<size_t,size_t>(es[i].t[0],es[i].count);
        }

        for (size_t i=0;i<fs.size();++i) {
            f_counts_[fs[i].t[0]] = fs[i].count;
            f_pairs[fs[i].t[0]] = std::pair<size_t,size_t>(fs[i].t[0],fs[i].count);
        }

        LOG(INFO) << "UnigramConstraint: obtaining ranks" << std::endl;
        // sort to get ranks
        std::sort(e_pairs.begin(),e_pairs.end(),misc::compare_pair_second);
        std::sort(f_pairs.begin(),f_pairs.end(),misc::compare_pair_second);

        // fill ranks into lookup array
        LOG(INFO) << "UnigramConstraint: filling ranks into array" << std::endl;
        for (size_t i=0;i<fs.size();++i) {
            f_ranks_[f_pairs[i].first] = i;
        }
        f_ranks_max_ = fs.size();

        for (size_t i=0;i<es.size();++i) {
            e_ranks_[e_pairs[i].first] = i;
        }
        e_ranks_max_ = es.size();

        // print info
        if (verbose) {
            for (size_t i=0;i<fs.size();++i) LOG(INFO) << f.getVocabConst().getWord(i) << " count=" << f_counts_[i] << " rank=" << f_ranks_[i] << std::endl;
            for (size_t i=0;i<es.size();++i) LOG(INFO) << e.getVocabConst().getWord(i) << " count=" << e_counts_[i] << " rank=" << e_ranks_[i] << std::endl;
        }

    }

    ConstraintAcceptCode accept(size_t e, size_t f, const Mapping & mapping, const partial_hyp_body & body, partial_hyp_head & head) {
        (void) mapping; // unused
        (void) body;
        (void) head;
        if (mapping.getCardinality()>int(card_max_)) return DONT_CARE;
        size_t cnt_diff = std::abs(int(e_counts_[e])-int(f_counts_[f]));
        double rank_diff = std::abs(double(e_ranks_[e])/double(e_ranks_max_)-double(f_ranks_[f])/double(f_ranks_max_));
        if (cnt_diff > cnt_diff_max_) return REJECT;
        if (rank_diff > rank_diff_max_) return REJECT;
        return DONT_CARE;
    }
private:
    const Counts & e_;
    const Counts & f_;
    std::vector<size_t> e_counts_, f_counts_;
    std::vector<size_t> e_ranks_, f_ranks_;
    size_t e_ranks_max_;
    size_t f_ranks_max_;
    double rank_diff_max_;
    size_t cnt_diff_max_;
    size_t card_max_;
};

class CoocConstraint : public Constraint {
public:
    CoocConstraint(const Cooc & e, const Cooc & f, bool verbose=false) : e_cooc_(e), f_cooc_(f), verbose_(verbose) {
        LOG(INFO) << "CoocConstraint: setting up vectors" << std::endl;
        LOG(INFO) << "CoocConstraint: filling vectors" << std::endl;
    }

    ConstraintAcceptCode accept(size_t e, size_t f, const Mapping & mapping, const partial_hyp_body & body, partial_hyp_head & head) {

        (void) body;
        (void) head;

        if (verbose_) LOG(INFO) << "check " << f_cooc_.getVocabConst().getWord(f) << " vs " << e_cooc_.getVocabConst().getWord(e) << std::endl;

        const std::set<word_id_t> * f_coocs = f_cooc_.getCooc(f);
        const std::set<word_id_t> * e_coocs = e_cooc_.getCooc(e);

        CHECK_NOTNULL(f_coocs);
        CHECK_NOTNULL(e_coocs);

        // then only words are candidates that do cooccur with all translated coocs
        for (std::set<word_id_t>::const_iterator it=f_coocs->begin();it!=f_coocs->end();++it) {
            word_id_t f_prime = *it;
            word_id_t e_prime = mapping.getE(f_prime);
            if (e_prime == VOCAB_WORD_NOT_FOUND || e_prime == MAPPING_NOT_MAPPED) continue;
            if (e_coocs->count(e_prime) == 0) {
//                LOG(INFO) << "REJECT because F COOC " << f_cooc_.getVocabConst().getWord(f_prime) << " maps to " << e_cooc_.getVocabConst().getWord(e_prime);
//                LOG(INFO) << " which is not a COOC of " << e_cooc_.getVocabConst().getWord(e) << std::endl;
//                LOG(INFO) << std::endl;
//                LOG(INFO) << "reject" << std::endl;
                return REJECT;
            }
        }

        return DONT_CARE;
    }

private:
    const Cooc & e_cooc_;
    const Cooc & f_cooc_;
    const bool verbose_;
};

// constraint based on reference mapping
class ReferenceConstraint : public Constraint {
public:
    ReferenceConstraint(const ReferenceMapping & ref_map, bool verbose=false) : ref_map_(ref_map), verbose_(verbose) {
        LOG(INFO) << "ReferenceConstraint: init" << std::endl;
    }

    ConstraintAcceptCode accept(size_t e, size_t f, const Mapping & mapping, const partial_hyp_body & body, partial_hyp_head & head) {

        (void) body;
        (void) head;
        (void) mapping;

        if (ref_map_.isSpecified(f)) {
          if (ref_map_.isCorrect(e,f)) {
            return FORCE_KEEP;
          } else {
            return REJECT;
          }
        }

        return DONT_CARE;
    }

private:
    const ReferenceMapping & ref_map_;
    const bool verbose_;
};

#endif /* CONSTRAINT_H_ */
