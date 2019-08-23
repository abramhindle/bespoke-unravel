#ifndef CHUNK_BEAM_H_
#define CHUNK_BEAM_H_

#include "global.hh"
#include "vocab.hh"
#include "kenlm_with_vocab.hh"
#include "misc.hh"
#include <vector>
#include <boost/iostreams/filtering_stream.hpp>
#include <set>

class ChunkBeam {
 public:
  void init(const Lexicon &lex);
  std::set<word_id_t> getCandidates(const std::vector<word_id_t> &input_sentence) const;

  ChunkBeam(Vocab &vocab,
            const std::vector<std::vector<word_id_t>> &sent,
            size_t windowSize, size_t beamSize)
      : vocab_(vocab),
        windowSize_(windowSize),
        beamSize_(beamSize) {
        // not sure about this
        addChunks(sent, 0, 0, &chunks_);
  };

  Vocab &getVocab() const {
    return vocab_;
  }

  virtual ~ChunkBeam() {};

 private:
  void addChunks(const std::vector<std::vector<word_id_t>> &sent,
                 word_id_t null_left, word_id_t null_right,
                 std::map<std::vector<word_id_t>, size_t> *chunks);

  Vocab &vocab_;
  size_t windowSize_, beamSize_;
  std::map<std::vector<word_id_t>, size_t> chunks_;
  std::map<std::vector<word_id_t>, std::vector<word_id_t>> candidates_;
};

#endif /* CHUNK_BEAM_H_ */
