#ifndef RESOURCE_H_
#define RESOURCE_H_

#include "config.h"

#include <fstream>

#include "mapping.hh"
#include "cooc.hh"
#include "kenlm_with_vocab.hh"
#include "counts.hh"
#include "classes.hh"
#include "embedding.hh"
#include "extension_order.hh"

class Resource {

public:
    virtual ~Resource() {
    }
};

class LmResource : public Resource {
    public:

    LmResource(std::string filename, Vocab * vocab) : filename_(filename), vocab_(vocab) {
        lm_.read(filename_.c_str(),vocab_);
    }

    Vocab * getVocab() {
        return vocab_;
    }

    KenLmWithVocab * getLm() {
        return &lm_;
    }

    private:
    std::string filename_;
    KenLmWithVocab lm_;
    Vocab * vocab_;
};

class CountsResource : public Resource {
    public:

    CountsResource(std::string filename, order_t order, count_t mincount, Vocab * vocab, bool pad_sent_start) : filename_(filename), order_(order), counts_(pad_sent_start), vocab_(vocab) {
        if (vocab_==0) vocab_ = new Vocab();
        // maybe skip this in the future
        bool calc_word_to_ngrams = true;
        counts_.read(filename, (order_t) order, mincount, *vocab_, calc_word_to_ngrams);
        counts_.printNumberOfNgrams();
    }

    Vocab * getVocab() {
        return vocab_;
    }

    Counts * getCounts() {
        return &counts_;
    }

    order_t getOrder() {
      return order_;
    }

    private:
    std::string filename_;
    order_t order_;
    Counts counts_;
    Vocab * vocab_;
};

class ClassesResource : public Resource {
    public:

    ClassesResource(std::string filename, Vocab * vocab, bool verbose) : filename_(filename), vocab_(vocab) {
        if (vocab_==0) vocab_ = new Vocab();
        std::ifstream classesFile(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        boost::iostreams::filtering_istream classesFileIn;
        classesFileIn.push(boost::iostreams::gzip_decompressor());
        classesFileIn.push(classesFile);
        classes_.read(classesFileIn, *vocab_, verbose);
    }

    Vocab * getVocab() {
        return vocab_;
    }

    Classes * getClasses() {
        return &classes_;
    }

    private:
    std::string filename_;
    Classes classes_;
    Vocab * vocab_;
};

class CoocResource : public Resource {
    public:

    CoocResource(std::string filename, Vocab * vocab, bool verbose) : filename_(filename), vocab_(vocab) {
        if (vocab_==0) vocab_ = new Vocab();
        std::ifstream coocFile(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        boost::iostreams::filtering_istream coocFileIn;
        coocFileIn.push(boost::iostreams::gzip_decompressor());
        coocFileIn.push(coocFile);
        cooc_.read(coocFileIn, *vocab_, verbose);
    }

    Vocab * getVocab() {
        return vocab_;
    }

    Cooc * getCooc() {
        return &cooc_;
    }

    private:
    std::string filename_;
    Cooc cooc_;
    Vocab * vocab_;
};

class EmbeddingResource : public Resource {
    public:

    EmbeddingResource(std::string filename, Vocab * vocab, size_t k, bool add_new, bool verbose) : filename_(filename), vocab_(vocab) {
        if (vocab_==0) vocab_ = new Vocab();
        embedding_.read(filename, *vocab_, add_new, verbose, k);
    }

    Vocab * getVocab() {
        return vocab_;
    }

    Embedding * getEmbedding() {
        return &embedding_;
    }

    private:
    std::string filename_;
    Embedding embedding_;
    Vocab * vocab_;
};

class VocabResource : public Resource {
    public:
    VocabResource(Vocab * vocab) : vocab_(vocab) {
    }

    Vocab * getVocab() {
        return vocab_;
    }

    private:
    Vocab * vocab_;
};

class ReferenceMappingResource : public Resource {

    public:
    ReferenceMappingResource(ReferenceMapping * referenceMapping) : referenceMapping_(referenceMapping) {
    }

    ReferenceMapping * getReferenceMapping() {
        return referenceMapping_;
    }

    private:
    ReferenceMapping * referenceMapping_;
};

#endif /* RESOURCE_H_ */
