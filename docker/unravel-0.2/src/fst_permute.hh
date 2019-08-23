// permute.h

#ifndef FST_PERMUTE_H__
#define FST_PERMUTE_H__

#include <algorithm>
#include <vector>
#include <map>
#include <cstdint>
#include <fst/types.h>
#include <fst/symbol-table.h>
#include <fst/weight.h>
#include <fst/test-properties.h>
#include <fst/mutable-fst.h>
#include <fst/rational.h>

namespace fst {

    // bitvector class from fsa's implementation of permutation automata
    class Bitvector {
        private:
            size_t size_;
            std::vector<std::uint32_t> bits_;
        public:
            Bitvector(const Bitvector &v, size_t flipPos) :
                size_(v.size_), bits_(v.bits_) {
                    flip(flipPos);
                }
            Bitvector(size_t size, bool value = false) :
                size_(size), bits_((size + 31) / 32, value ? ~std::uint32_t(0) : 0) {
                }
            Bitvector() : size_(0) {}
            size_t size() const {
                return size_;
            }
            bool operator[](size_t pos) const {
                return (bits_[pos / 32] >> (pos % 32)) & 1;
            }
            bool operator==(const Bitvector &v) const {
                return (bits_ == v.bits_);
            }
            bool operator<(const Bitvector& v) const {
                return bits_ < v.bits_;
            }

            void flip(size_t pos) {
                bits_[pos / 32] ^= (1 << (pos % 32));
            }
            std::uint32_t hashKey() const {
                std::uint32_t value = 0;
                for (std::vector<std::uint32_t>::const_iterator i = bits_.begin(); i
                        != bits_.end(); ++i)
                    value |= *i;
                return value;
            }

            static void printBin(std::ostream& out, std::uint32_t x) {
                // print std::uint32_t in readable form, usefull for debugging
                for (int i = 0; i < 32; ++i)
                    out << ((x >> i) & 1 ? "1" : "0");
            }

            bool isCovered(size_t A, size_t B) const {
                // check if range [A,B) is covered

                size_t Abi = A / 32, Bbi = B / 32;
                for (size_t i = Abi + 1; i < Bbi; ++i)
                    if (!(bits_[i] == ~std::uint32_t(0)))
                        return 0;

                std::uint32_t maskA = (~std::uint32_t(0) >> (A % 32)) << (A % 32);
                std::uint32_t maskB = (~std::uint32_t(0) << (32 - (B % 32))) >> (32 - (B % 32));

                if (Abi == Bbi)
                    return (bits_[Abi] & (maskA & maskB)) == (maskA & maskB);
                else
                    return ((bits_[Abi] & maskA) == maskA) && ((bits_[Bbi] & maskB)
                            == maskB);
            }

            friend std::ostream& operator<<(std::ostream &o, const Bitvector &b) {
                for (std::uint32_t i = 0; i < b.size_; ++i)
                    o << (b[i] ? "1" : "0");
                return o;
            }
    };

    // state of the permutation automaton (coverage and depth)
    struct PermutationState_ {
        std::uint32_t depth_;
        Bitvector used_;
        size_t id_;
        PermutationState_() {}
        PermutationState_(std::uint32_t depth, const Bitvector &used) : depth_(depth), used_(used) {
        }
        bool operator<(const PermutationState_ &s) const {
            return (used_ < s.used_);
        }
        bool operator==(const PermutationState_ &s) const {
            return (used_ == s.used_);
        }
        std::uint32_t hashKey() const {
            return 2239 * depth_ + used_.hashKey();
        }
    };

    // implementation of permutation automaton
    template<class FstArc> class PermutationAutomatonImpl: public fst::FstImpl<FstArc> {
        typedef fst::Fst<FstArc> FstAutomaton;
        typedef FstArc Arc;
        typedef typename FstArc::Weight Weight;
        typedef typename FstArc::StateId StateId;
        using FstImpl<Arc>::SetType;
        using FstImpl<Arc>::SetProperties;
        using FstImpl<Arc>::SetInputSymbols;
        using FstImpl<Arc>::SetOutputSymbols;

        public:

        PermutationAutomatonImpl(FstAutomaton * fst, int windowSize, double forwardProb) : fst_(fst), windowSize_(windowSize), forwardProb_(forwardProb), margin_(1) {
            // TODO: find out number of states
            SetType("permutation");
            SetInputSymbols(fst_->InputSymbols());
            SetOutputSymbols(fst_->OutputSymbols());
            int numArcs = checkLinearity();
            if (numArcs >= 0) {
//                std::cerr << "adding first state" << std::endl;
                PermutationState_ state(0,Bitvector(numArcs));
                permutationStates_[state.hashKey()] = state;
                if (fstStateToHash_.find(state.hashKey()) == fstStateToHash_.end()) fstStateToHash_[state.hashKey()] = fstStateToHash_.size();
            }
//            std::cerr << "num arcs " << numArcs << std::endl;
        }

        virtual ~PermutationAutomatonImpl() {
        }

        virtual StateId Start() const {
//            std::cerr << "Start()" << std::endl;
            return 0;
        }

        virtual Weight Final(StateId id) {
            if (expandState(id) > 0) {
                return Weight::Zero();
            } else {
                return Weight::One();
            }
        }
        virtual size_t NumArcs(StateId id) {
            size_t result = expandState(id);
//            std::cerr << "NumArcs(" << id << ") = " << result << std::endl;
            return result;
        }
        virtual size_t NumInputEpsilons(StateId id) const {
            (void) id;
            return 0;
        }
        virtual size_t NumOutputEpsilons(StateId id) const {
            (void) id;
            return 0;
        }
        FstAutomaton* getFst() const {
            return fst_;
        }
        // return false if state not yet initialized
        bool validState(StateId state) {
//            std::cerr << "validState(" << state << ")" << std::endl;
            return fstStateToHash_.find(state) != fstStateToHash_.end();
        }
        // expand state and return number of arcs
        int expandState(StateId state, std::vector<Arc> * arcs = NULL) {
//            std::cerr << "call: expandState(" << state << ")" << std::endl;
            state = fstStateToHash_[state];
//            std::cerr << "converted id to hash " << state << std::endl;
            if (permutationStates_.find(state) == permutationStates_.end()) {
//                std::cerr << "expandState: state not yet initialized" << std::endl;
                return -1;
            }

            // find first bit not set, and iterate on original automaton
            std::uint32_t firstBitNotSet;

            // at the same time iterate over the fst
            fst::StateIterator<FstAutomaton> siter(*fst_);
            for (firstBitNotSet = 0; permutationStates_[state].used_[firstBitNotSet]; firstBitNotSet++, siter.Next());

            // find upper limit
            std::uint32_t upperLimit = std::min(size_t(firstBitNotSet + windowSize_), permutationStates_[state].used_.size()-1);

/*            if (state==0) {
                std::cerr << "setting upper limit because of state==0" << std::endl;
                upperLimit = firstBitNotSet + 1;
            }*/

            if (firstBitNotSet >= upperLimit) {
                upperLimit = permutationStates_[state].used_.size();
            }

//            std::cerr << "upper limit = " << upperLimit << std::endl;

            //typename std::vector<Arc>::iterator fstArc = arcs_.begin();
            //arcs_.push_back(newArc);
            //
            std::uint32_t numArcs = 0;

//            std::cerr << "expandState: " << permutationStates_[state].depth_ << " - " <<
//                permutationStates_[state].used_ << " : id=" << state << " in range " << "[" << firstBitNotSet << "," << upperLimit << "]" << std::endl;

            // iterate i, outputArcs and on states of original fst
            for (std::uint32_t i = firstBitNotSet; i < upperLimit; ++i, siter.Next()) {
                if (!permutationStates_[state].used_[i]) {
                    numArcs+=1;
                    //std::uint32_t oldSize = permutationStates_.size();
                    std::uint32_t newDepth = permutationStates_[state].depth_ + 1;
                    Bitvector newBitvector = Bitvector(permutationStates_[state].used_, i);
                    PermutationState_ newState(newDepth, newBitvector);
                    std::uint32_t newStateHash = newState.hashKey();
                    StateId newStateId = 0;
                    int jump_width = i - firstBitNotSet;
                    // in window = 2 case, check if we need to go back
                    bool goback = permutationStates_[state].used_[upperLimit-1];

                    // ADD STATE //////////////////////////////////////////////////////////////////////////////////////////
                    if (permutationStates_.find(newStateHash) != permutationStates_.end()) {
                        newStateId = permutationStates_[newStateHash].id_;
                    } else {
                        newStateId = permutationStates_.size();
                        fstStateToHash_[newStateId] = newStateHash;
                    }
                    fstStateToHash_[newStateId] = newStateHash;
                    newState.id_ = newStateId;
                    permutationStates_[newStateHash] = newState;
                    ///////////////////////////////////////////////////////////////////////////////////////////////////////

//                    std::cerr << ">>new state: " << newDepth << " - " << newBitvector << " : hash=" << newStateHash << " id=" << newStateId << std::endl;

                    if (arcs != NULL) {
                        fst::ArcIterator<FstAutomaton> aiter(*fst_, siter.Value());
                        //std::cerr << "adding arc " << fst_->InputSymbols()->Find(aiter.Value().ilabel) << std::endl;
                        Arc newArc;
                        newArc.ilabel = aiter.Value().ilabel;
                        newArc.olabel = aiter.Value().olabel;

                        // THIS ONLY WORDS FOR WINDOW 2
                        if (jump_width == 0) {
                            newArc.weight = -::log(forwardProb_); // TODO: this is not correct! should be -log(1.0) if there is just one arc
                        } else {
                            CHECK_EQ(windowSize_, 2);
                            newArc.weight = -::log(1.0-forwardProb_);
                        }

                        if (goback) newArc.weight = 0.0;

                        //newArc.weight = fst::Plus(aiter.Value().weight,Weight(jump_width)); // jump penalty
                        newArc.nextstate = newStateId;
                        arcs->push_back(newArc);
//                        std::cerr << "adding arc: from" << state << " to " << newStateId << std::endl;
                    }

                }
            }
//            std::cerr << "returning " << numArcs << " arcs" << std::endl;
            return numArcs;
        }

        private:

        // check if automaton is linear
        int checkLinearity() {
//            std::cerr << "checkLinearity()" << std::endl;
            int num_states = 0;
            for (fst::StateIterator<FstAutomaton> siter(*fst_); !siter.Done(); siter.Next(), ++num_states) {
                const int q = siter.Value();
                int num_arcs = 0;
                for (fst::ArcIterator<FstAutomaton> aiter(*fst_, q); !aiter.Done(); aiter.Next(), ++num_arcs);
                if (num_arcs > 1) {
//                    std::cerr << "automaton is not linear" << std::endl;
                    return -1;
                }
            }
            return (num_states-1);
        }

        std::map<std::uint32_t,PermutationState_> permutationStates_;
        std::map<StateId,std::uint32_t> fstStateToHash_;
        const FstAutomaton *fst_;
        int windowSize_;
        double forwardProb_;
        int margin_;
    };

    // arc iterator
    template <class F> class PermutationAutomatonArcIteratorBase : public fst::ArcIteratorBase<typename F::Arc> {
        public:
            typedef typename F::Arc Arc;
            typedef typename Arc::Weight Weight;
            typedef typename Arc::StateId StateId;
            typedef typename F::FstAutomaton FstAutomaton;

            explicit PermutationAutomatonArcIteratorBase(PermutationAutomatonImpl<typename F::Arc> *impl, StateId state) : impl_(impl), i(0) {
                // expand state and get arcs
                impl_->expandState(state, &arcs_);
            }

            virtual ~PermutationAutomatonArcIteratorBase() {}
        private:
            virtual bool Done_() const { return (i >= arcs_.size()); }
            virtual const Arc& Value_() const { return arcs_[i]; }
            virtual void Next_() { ++i; }
            virtual size_t Position_() const { return i; }
            virtual void Reset_() { i = 0; }
            virtual void Seek_(size_t a) { i = a; }
            virtual uint32 Flags_() const { return fst::kArcValueFlags; }
            virtual void SetFlags_(uint32 flags, uint32 mask) {
                (void) flags;
                (void) mask;
            }
        private:
            PermutationAutomatonImpl<typename F::Arc> * impl_;
            std::vector<Arc> arcs_;
            size_t i;
    };


    // state iterator
    template <class F> class PermutationAutomatonStateIteratorBase : public fst::StateIteratorBase<typename F::Arc> {
        public:
            typedef typename F::StateId StateId;
            explicit PermutationAutomatonStateIteratorBase(PermutationAutomatonImpl<typename F::Arc> *impl) : impl_(impl) , state_(0) {}
            virtual ~PermutationAutomatonStateIteratorBase() {}
        private:
            virtual bool Done_() const {
//                std::cerr << "Done()?" << state_ << std::endl;
                return !(impl_->validState(state_));
            }
            virtual StateId Value_() const { 
//                std::cerr << "state iterator: value " << state_ << std::endl;
                return state_;
            }
            virtual void Next_() {
                impl_->expandState(state_);
//                std::cerr << "next" << std::endl;
                ++state_;
            }
            virtual void Reset_() { state_ = 0; }
            PermutationAutomatonImpl<typename F::Arc> * impl_;
            int state_;
    };

    // permutation automaton
    template<class FstArc> class PermutationAutomaton: public fst::Fst<FstArc> {
        public:
            typedef fst::Fst<FstArc> FstAutomaton;
            typedef fst::FstImpl<FstArc> Predecessor;
            typedef PermutationAutomatonImpl<FstArc> Impl;
            typedef PermutationAutomaton<FstArc> Self;
            typedef FstArc Arc;
            typedef typename FstArc::Weight Weight;
            typedef typename FstArc::StateId StateId;

            PermutationAutomaton(FstAutomaton * fst, int windowSize, double forwardProb) : impl_(new Impl(fst,windowSize,forwardProb)) {
            }

            PermutationAutomaton(const Self &fst, bool reset) {
                if (reset) {
                    impl_ = new Impl(*fst.impl_);
                } else {
                    impl_ = fst.impl_;
                    impl_->IncrRefCount();
                }
            }

            virtual ~PermutationAutomaton() {
                if (!impl_->DecrRefCount())
                    delete impl_;
            }
            virtual StateId Start() const {
                return impl_->Start();
            }
            virtual Weight Final(StateId id) const {
                return impl_->Final(id);
            }
            virtual size_t NumArcs(StateId id) const {
                return impl_->NumArcs(id);
            }
            virtual size_t NumInputEpsilons(StateId id) const {
                return impl_->NumInputEpsilons(id);
            }
            virtual size_t NumOutputEpsilons(StateId id) const {
                return impl_->NumOutputEpsilons(id);
            }
            virtual uint64 Properties(uint64 mask, bool test) const {
                if (test) {
                    uint64 knownprops, testprops = TestProperties(*this, mask, &knownprops);
                    //impl_->SetProperties(testprops, knownprops);
                    return testprops & mask;
                } else {
                    return impl_->Properties(mask);
                }
            }
            virtual const string& Type() const {
                return impl_->Type();
            }
            virtual Self* Copy(bool reset = false) const {
                return new Self(*this, reset);
            }
            virtual const fst::SymbolTable* InputSymbols() const {
                return impl_->InputSymbols();
            }
            virtual const fst::SymbolTable* OutputSymbols() const {
                return impl_->OutputSymbols();
            }
            virtual void InitStateIterator(fst::StateIteratorData<FstArc> *data) const {
                data->base = new PermutationAutomatonStateIteratorBase<Self> ( impl_ );
            }
            virtual void InitArcIterator(StateId s, fst::ArcIteratorData<FstArc> *data) const {
                data->base = new PermutationAutomatonArcIteratorBase<Self> ( impl_, s );
            }

        private:
            Impl *impl_;
    };
}

#endif  // FST_PERMUTE_H__
