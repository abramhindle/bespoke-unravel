#ifndef FST_BEAMSEARCH_HH
#define FST_BEAMSEARCH_HH

#include <fst/connect.h>
#include <fst/fst.h>
#include <fst/reverse.h>
#include <vector>
#include <unordered_map>

namespace {
    template<class A>
        struct ShortestPathHyp {
            typename A::Weight weight;
            typename A::StateId state, trace;
        };
}

/**
 * Arc-synchronous breadth-first beam search
 */
template<class A> void shortestPaths(const fst::Fst<A> &fst, typename A::Weight beam, fst::VectorFst<A> *lattice) {
    typedef fst::Fst<A> Fst;
    typedef typename A::StateId StateId;
    typedef typename A::Weight Weight;
//    typedef typename A::Label Label;
    typedef ShortestPathHyp<A> Hyp;
    std::vector<Hyp> active, newActive;
    CHECK(Weight::Properties() & fst::kPath);
    fst::VectorFst<A> traceback;
    typedef std::unordered_map<StateId, size_t> StateToHypMap;
    StateToHypMap stateToHyp;
    lattice->DeleteStates();
    Hyp start;
    start.state = fst.Start();
    start.weight = Weight::One();
    start.trace = traceback.AddState();
    traceback.SetStart(start.trace);
    active.push_back(start);
    Hyp bestFinal;
    bestFinal.state = fst::kNoStateId;
    bestFinal.weight = Weight::Zero();
    bestFinal.trace = traceback.AddState();
    while (!active.empty() && bestFinal.state == fst::kNoStateId) {
        std::cerr << "active.size=" << active.size() << std::endl;
        // expand active hypotheses
        Weight best = Weight::Zero();
        size_t i=0;
        for (typename std::vector<Hyp>::const_iterator hyp = active.begin(); hyp != active.end(); ++hyp) {
            std::cerr << "working on active[" << i++ << "] -  state=" << hyp->state << std::endl;
            // determine best active final state if any
            if (fst.Final(hyp->state) != Weight::Zero()) {
                Weight finalWeight = fst::Times(hyp->weight, fst.Final(hyp->state));
                if (bestFinal.weight != fst::Plus(bestFinal.weight, finalWeight)) {
                    bestFinal.weight = fst::Plus(bestFinal.weight, finalWeight);
                    bestFinal.state = hyp->state;
                }
                traceback.AddArc(bestFinal.trace, A(0, 0, fst.Final(hyp->state), hyp->trace));
            }

            std::cerr << "expand arcs.." << std::endl;
            // expand hypotheses
            for (fst::ArcIterator<Fst> aiter(fst, hyp->state); !aiter.Done(); aiter.Next()) {
                Hyp newHyp;
                A arc = aiter.Value();
                newHyp.state = arc.nextstate;
                newHyp.weight = fst::Times(hyp->weight, arc.weight);
                //std::cerr << "expanding arc to state " << arc.nextstate << " with weight " << arc.weight << std::endl;
                typename StateToHypMap::const_iterator i = stateToHyp.find(newHyp.state);
                if (i != stateToHyp.end()) {
                    // recombine hypotheses
                    Hyp &h = newActive[i->second];
                    newHyp.trace = h.trace;
                    if (h.weight != fst::Plus(h.weight, newHyp.weight)) {
                        h = newHyp;
                    }
                } else {
                    stateToHyp.insert(typename StateToHypMap::value_type(newHyp.state, newActive.size()));
                    newHyp.trace = traceback.AddState();
                    newActive.push_back(newHyp);
                }
                // add backpointers
                arc.nextstate = hyp->trace;
                traceback.AddArc(newHyp.trace, arc);

                // keep track of the best score (for pruning)
                if (best != fst::Plus(best, newHyp.weight))
                    best = fst::Plus(best, newHyp.weight);
            } // for aiter
            std::cerr << "expand arcs..done" << std::endl;
        } // for hyp
        // prune hypotheses
        active.clear();
        Weight threshold = fst::Times(best, beam);
        std::cerr << "pruning: best=" << best << " threshold=" << threshold << std::endl;
        for (typename std::vector<Hyp>::const_iterator hyp = newActive.begin(); hyp != newActive.end(); ++hyp) {
            if (threshold != fst::Plus(hyp->weight, threshold)) {
                active.push_back(*hyp);
            }
        }
        /*! @todo garbage collection in the traceback: remove non-reachable states */
        newActive.clear();
        stateToHyp.clear();
    } // while
    if (bestFinal.state != fst::kNoStateId) {
        // create lattice
        traceback.SetFinal(traceback.Start(), Weight::One());
        traceback.SetStart(bestFinal.trace);
        fst::Connect(&traceback);
        fst::Reverse(traceback, lattice);
    } else {
        // no final state found
    }
}

#endif // FST_BEAMSEARCH_HH
