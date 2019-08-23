AM_CPPFLAGS = -I.. $(BOOST_CPPFLAGS)

if WITH_FIXUBUNTU
NO_AS_NEEDED = -Wl,--no-as-needed
endif

LIBTEMPLATE=libtemplate.la

noinst_LTLIBRARIES = libtemplate.la
noinst_PROGRAMS = template_test template_main
noinst_HEADERS = src/template.hh
TEST = template_test

libtemplate_la_CXXFLAGS = -O3
libtemplate_la_SOURCES = src/template.cc \
                        src/template.hh

template_test_SOURCES = src/template_test.cc src/template.hh src/template.cc
template_test_CXXFLAGS = -pthread $(NO_AS_NEEDED)
template_test_LDADD = $(LIBTEMPLATE) -lgtest -lglog -lgtest_main

template_main_LDADD = $(LIBTEMPLATE) -lgflags -lglog -lboost_system -lboost_program_options
template_main_SOURCES = src/template_main.cc \
                   src/template.cc \
                   src/template.hh
template_main_CXXFLAGS = ${OPENMP_FLAG} -Wall -Wextra -Wunused-parameter -g -march=native -O3

noinst_DATA = .gitignore

.gitignore: Makefile.am
	cat $(top_builddir)/.gitignore > .gitignore
	for bin in $(PROGRAMS); do \
	  echo $$bin >> .gitignore; \
	done; \
	for bin in $(SNAPPY_MAIN); do \
	  echo $$bin >> .gitignore; \
	done; \
	for bin in $(SCRIPTS); do \
	  echo $$bin >> .gitignore; \
	done; \
	for bin in $(IGNORES); do \
	  echo $$bin >> .gitignore; \
	done
