ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

if WITH_OPENMP
OPENMP_FLAG=-fopenmp
endif

if WITH_FIXUBUNTU
NO_AS_NEEDED = -Wl,--no-as-needed -lrt
endif

# -Wfloat-equal produces warnings from kenlm headers
# -Wconversion -Wsign-conversion causes many warnings
# -Wunsafe-loop-optimizations causes many warnings at high optimization level
# -Wmissing-include-dirs

GIT_REVISION=$(shell git describe --abbrev=4 --dirty --always --tags)
WARNING_CXXFLAGS = -Wall -Wextra -Wunused-parameter
DEBUG_CXXFLAGS=-g -O3 #-pg
AM_CXXFLAGS=-DGIT_REVISION=\"$(GIT_REVISION)\" $(OPENMP_FLAG) $(DEBUG_CXXFLAGS) $(WARNING_CXXFLAGS) -march=native # -Wa,-q # last is for mac
AM_LDFLAGS=$(NO_AS_NEEDED)
GOOGLE_LDADD=-lgtest -lgflags -lglog

EXTRA_DIST=\
					 data/tests/det_z408/cipher.*gram.gz \
					 data/tests/det_z408/download_lm.sh \
					 data/tests/det_z408/run.sh \
					 data/tests/det_z408/flagfile* \
					 data/tests/em_1to1/e \
					 data/tests/em_1to1/f \
					 data/tests/em_1to1/download_lm.sh \
					 data/tests/em_1to1/run.sh \
					 data/tests/em_1to1/flagfile* \
					 src/cdec/LICENSE.txt \
					 src/cdec/mteval/*.cc \
					 src/cdec/mteval/*.h \
					 src/cdec/utils/*.cc \
					 src/cdec/utils/*.h \
					 codec/Makefile.am \
					 codec/*.c \
					 codec/*.h \
					 codec/*.cpp \
					 codec/*.hpp \
					 codec/utils/* \
					 template/Makefile.am \
					 template/src/*.cc \
					 template/src/*.hh \
					 .gitignore

noinst_PROGRAMS = \
									vocab_test \
									lexicon_test

noinst_LIBRARIES = \ 
									libkenlm.a \
									libfeatures.a \
									libscore.a

noinst_LTLIBRARIES = \
									libnlp.la \
									libmisc.la

bin_PROGRAMS = \
									det_unravel \
									sparse \
									fast_score \
									em_unravel

TEST = \
									vocab_test \
									lexicon_test

LIBMISC = $(top_builddir)/libmisc.la $(srcdir)/codec/libcodec.la
LIBSCORE = $(top_builddir)/libscore.a
LIBKENLM = $(top_builddir)/libkenlm.a
LIBFEATURES = $(top_builddir)/libfeatures.a
LIBNLP = $(top_builddir)/libnlp.la $(LIBMISC)

libnlp_la_CXXFLAGS =
libnlp_la_LIBADD = -lfst $(GOOGLE_LDADD) -lboost_program_options -lboost_system -lboost_thread -lboost_chrono -lz
libnlp_la_SOURCES = \
  src/extension_order.cc \
  src/extension_order.hh \
  src/counts.cc \
  src/counts.hh \
  src/classes.cc \
  src/classes.hh \
  src/cooc.cc \
  src/cooc.hh \
  src/embedding.cc \
  src/embedding.hh \
  src/mapping.cc \
  src/mapping.hh \
  src/reference_mapping.cc \
  src/reference_mapping.hh \
  src/lexicon.hh \
  src/lexicon_sparse.cc \
  src/lexicon_sparse.hh \
  src/vocab.cc \
  src/vocab.hh \
	src/murmur_hash.h \
	src/have_64_bits.h

libmisc_la_CXXFLAGS =
libmisc_la_LIBADD =
libmisc_la_SOURCES = \
	src/misc.cc \
	src/misc.hh \
	src/misc_io.hh \
	src/misc_io.cc \
	src/misc_time.hh \
	src/misc_time.cc \
	src/misc_prettyprint.hh \
	src/misc_prettyprint.cc

lexicon_test_SOURCES = src/lexicon_test.cc \
                       src/lexicon_sparse.hh \
                       src/lexicon_sparse.cc
lexicon_test_LDADD = $(LIBMISC) $(LIBNLP) $(LIBKENLM) $(LIBMISC)
lexicon_test_LDFLAGS = $(AM_LDFLAGS) -ldl -lgtest -pthread -lgtest_main -lkenlm

vocab_test_SOURCES = src/vocab_test.cc src/vocab.hh src/vocab.cc
vocab_test_LDADD = $(LIBMISC) $(LIBNLP)
vocab_test_LDFLAGS = $(AM_LDFLAGS) -ldl -lgtest -pthread -lgtest_main

libkenlm_a_SOURCES = \
	src/vocab.hh \
	src/lm.hh \
	src/automaton_lm.hh \
	src/automaton_lm.cc \
	src/kenlm_with_vocab.hh \
	src/kenlm_with_vocab.cc \
	src/actual_strings_kenlm.cc \
	src/actual_strings_kenlm.hh

libfeatures_a_SOURCES = \
	src/feature.hh \
	src/ngram_feature.cc \
	src/ngram_feature.hh \
	src/oracle_feature.cc \
	src/oracle_feature.hh \
	src/cooc_feature.cc \
	src/cooc_feature.hh \
	src/embedding_feature.cc \
	src/embedding_feature.hh \
	src/phrase_feature.cc \
	src/phrase_feature.hh \
  src/multi_ngram_feature.cc \
	src/multi_ngram_feature.hh

libscore_a_SOURCES = \
	src/cdec/utils/stringlib.cc \
	src/cdec/utils/stringlib.h \
	src/cdec/utils/filelib.cc \
	src/cdec/utils/filelib.h \
	src/cdec/utils/tdict.cc \
	src/cdec/utils/tdict.h \
	src/cdec/utils/dict.cc \
	src/cdec/utils/dict.h \
	src/cdec/utils/wordid.h \
	src/cdec/utils/gzstream.cc \
	src/cdec/utils/gzstream.h \
	src/cdec/utils/null_deleter.h \
	src/cdec/utils/hash.h \
	src/cdec/utils/murmur_hash.h \
	src/cdec/utils/have_64_bits.h \
	src/cdec/mteval/ns.cc \
	src/cdec/mteval/ns.h \
	src/cdec/mteval/ns_docscorer.cc \
	src/cdec/mteval/ns_docscorer.h \
	src/cdec/mteval/ns_ter.cc \
	src/cdec/mteval/ns_ter.h \
	src/cdec/mteval/ns_cer.cc \
	src/cdec/mteval/ns_comb.cc \
	src/cdec/mteval/ns_ext.cc \
	src/cdec/mteval/ns_ssk.cc \
	src/cdec/mteval/ns_cer.h \
	src/cdec/mteval/ns_comb.h \
	src/cdec/mteval/ns_ext.h \
	src/cdec/mteval/ns_ssk.h

fast_score_SOURCES = src/fast_score.cc
fast_score_CXXFLAGS = $(AM_CXXFLAGS)
fast_score_LDFLAGS = $(AM_LDFLAGS)
fast_score_LDADD = $(LIBSCORE) $(LIBMISC) $(LIBNLP) $(GOOGLE_LDADD) -ldl -lfst -lz -lboost_program_options

sparse_SOURCES = src/sparse_main.cc \
	src/fst_beamsearch.hh \
	src/fst_permute.hh \
	src/lexicon.hh \
	src/vocab.hh \
	src/resource.hh \
	src/beam_search.cc \
	src/beam_search.hh \
	src/config_options.hh \
	src/counts.hh \
	src/classes.hh \
	src/cooc.hh \
	src/extension_order.hh \
	src/global.hh \
	src/constraint.hh \
	src/hypothesis_struct.hh \
	src/misc.hh \
	src/ngram.hh \
	src/mapping.hh \
	src/reference_mapping.hh \
	src/statistics.hh
sparse_CXXFLAGS = $(AM_CXXFLAGS)
sparse_LDFLAGS = $(AM_LDFLAGS)
sparse_LDADD = $(LIBSCORE) $(LIBMISC) $(LIBNLP) $(LIBKENLM) $(GOOGLE_LDADD) -ldl -lfst -lkenlm

em_unravel_SOURCES = src/em_unravel_main.cc \
	src/em_unravel_helpers.hh \
	src/beam_search.cc \
	src/beam_search.hh \
	src/lm_beam.cc \
	src/lm_beam.hh \
	src/chunk_beam.cc \
	src/chunk_beam.hh \
	src/fst_beamsearch.hh \
	src/fst_permute.hh \
	src/lexicon.hh \
	src/vocab.hh \
	src/resource.hh \
	src/config_options.hh \
	src/counts.hh \
	src/classes.hh \
	src/cooc.hh \
	src/extension_order.hh \
	src/global.hh \
	src/constraint.hh \
	src/hypothesis_struct.hh \
	src/misc.hh \
	src/ngram.hh \
	src/mapping.hh \
	src/reference_mapping.hh \
	src/statistics.hh
em_unravel_CXXFLAGS = $(AM_CXXFLAGS)
em_unravel_LDFLAGS = $(AM_LDFLAGS)
em_unravel_LDADD = $(LIBMISC) $(LIBNLP) $(LIBKENLM) $(LIBSCORE) $(GOOGLE_LDADD) -ldl -lfst -lboost_program_options -lboost_system -lboost_thread -lboost_chrono -lz -lkenlm

det_unravel_SOURCES = \
	src/global.hh \
	src/misc.hh \
	src/vocab.hh \
	src/resource.hh \
	src/beam_search.cc \
	src/beam_search.hh \
	src/config_options.hh \
	src/counts.hh \
	src/classes.hh \
	src/cooc.hh \
	src/embedding.hh \
	src/det_unravel_main.cc \
	src/extension_order.hh \
	src/constraint.hh \
	src/hypothesis_struct.hh \
	src/ngram.hh \
	src/mapping.hh \
	src/reference_mapping.hh \
	src/statistics.hh
det_unravel_CXXFLAGS = $(AM_CXXFLAGS)
det_unravel_LDFLAGS = $(AM_LDFLAGS)
det_unravel_LDADD = $(LIBKENLM) $(LIBMISC) $(LIBFEATURES) $(LIBNLP) $(GOOGLE_LDADD) -ldl -lboost_program_options -lboost_system -lboost_thread -lboost_chrono -lz -lkenlm

dist_noinst_SCRIPTS = autogen.sh

# -local targets don't override the original targets. They are called in addition
#  #  be careful: .deps is generated in src/gen-cpp by configure
clean-local:
	rm -rf src/gen-cpp/* src/gen-py/*
