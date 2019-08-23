#ifndef BEAM_SEARCH_HH_
#define BEAM_SEARCH_HH_

#include "global.hh"
#include "feature.hh"
#include "statistics.hh"
#include "mapping.hh"
#include "reference_mapping.hh"
#include "misc.hh"
#include "hypothesis_struct.hh"
#include "config_options.hh"
#include "constraint.hh"

#include <vector>
#include <iomanip>
#include <algorithm>

#ifdef WITH_OPENMP
#include <parallel/algorithm>
#include <omp.h>
#endif

#include <boost/atomic.hpp>

class BeamSearch {
public:
    BeamSearch(const Vocab& f_vocab, const Vocab& e_vocab, const std::vector<std::string>& special_words);
    virtual ~BeamSearch();
    std::vector<Mapping> search(const ExtensionOrder& extOrder, const
        std::vector<size_t>& limit_e, std::vector<Constraint*> constraints,
        std::vector<Feature*> features, const ReferenceMapping*
        referenceMapping, const std::vector<score_t>& pruneThresh, const
        std::vector<size_t>& pruneMinHyp ,const std::vector<size_t>& pruneHist,
        const std::vector<size_t>& prune_mapping, const std::vector<size_t>&
        prune_predecessor, const std::string& best_hyp_filename, bool
        printBest);
    void fastForward(const ExtensionOrder& extOrder, std::istream* is,
        std::vector<Feature*> features);
    void setSearchLog(option_searchlog::OPTION_SEARCHLOG option_searchlog,
        std::ostream* search_log_os=0);
    void unsetRunning();

private:
    const Vocab& f_vocab_;
    const Vocab& e_vocab_;
    const std::vector<std::string>& special_words_;
    option_searchlog::OPTION_SEARCHLOG option_searchlog_;
    std::ostream* search_log_os_;
    std::vector<partial_hyp_body> sourceVector;
    std::vector<partial_hyp_body> targetVector;
    std::vector<partial_hyp_head> hypVector;

    std::set<size_t> special_e_words;
    std::set<size_t> special_f_words;
    size_t current_card;
    size_t current_hyps;

    bool running;

    void add_fixed_words();
    void init_bodies();

};

#endif /* BEAM_SEARCH_HH_ */
