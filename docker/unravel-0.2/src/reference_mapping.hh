#ifndef REFERENCEMAPPING_H_
#define REFERENCEMAPPING_H_

#include "global.hh"
#include "vocab.hh"
#include "mapping.hh"
#include "extension_order.hh"
#include <boost/iostreams/filtering_stream.hpp>
#include <vector>
#include <set>
#include <map>

class Mapping;

class ReferenceMapping {
public:

    ReferenceMapping();
    virtual ~ReferenceMapping();
    void id(const Vocab& evocab, const Vocab& fvocab);
    void idUnderscore(const Vocab& evocab, const Vocab& fvocab);
    void read(boost::iostreams::filtering_istream& is,
              const Vocab& evocab, const Vocab& fvocab);
    bool isCorrect(word_id_t e, word_id_t f) const;
    bool isSpecified(word_id_t f) const;
    size_t correctCount(const Mapping& mapping) const;

private:
    std::map<word_id_t,std::set<word_id_t> > possibleEs_;
};

#endif /* REFERENCEMAPPING_H_ */
