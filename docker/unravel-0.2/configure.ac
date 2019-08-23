AC_INIT([Unravel], [0.2], [nuhn@cs.rwth-aachen.de],
        [unravel], [http://www-i6.informatik.rwth-aachen.de/unravel/])
AC_CONFIG_HEADERS([config.h])
AC_PREREQ([2.63])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([1.11 -Wall no-define])

AC_LANG([C++])
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AX_CXX_COMPILE_STDCXX_11([noext], [mandatory])

save_LIBS="$LIBS"

AC_ARG_WITH([openmp], AS_HELP_STRING([--without-openmp], [do not use openmp]))
AS_IF([test "x$with_openmp" != "xno"], [have_openmp=yes], [have_openmp=no])
AS_IF([test "x$have_openmp" == "xyes"], [AC_MSG_NOTICE([openmp will be used])], [AC_MSG_NOTICE([openmp will not be used])])
AS_IF([test "x$have_openmp" == "xyes"], [AC_DEFINE([WITH_OPENMP], [1], [Used in source files])])
AM_CONDITIONAL(WITH_OPENMP, test x$have_openmp = xyes)

AC_ARG_WITH([fixubuntu], AS_HELP_STRING([--without-fixubuntu], [do apply fixes for broken ubuntu pipeline]))
AS_IF([test "x$with_fixubuntu" != "xno"], [have_fixubuntu=yes], [have_fixubuntu=no])
AS_IF([test "x$have_fixubuntu" == "xyes"], [AC_MSG_NOTICE([ubuntu fixes will be used])], [AC_MSG_NOTICE([ubuntu fixes will not be used])])
AS_IF([test "x$have_fixubuntu" == "xyes"], [AC_DEFINE([WITH_FIXUBUNTU], [1], [Used in source files])])
AM_CONDITIONAL(WITH_FIXUBUNTU, test x$have_fixubuntu = xyes)

AC_ARG_WITH([openfst], AS_HELP_STRING([--without-openfst], [do not use openfst]))
AS_IF([test "x$with_openfst" != "xno"], [have_openfst=yes], [have_openfst=no])
AS_IF([test "x$have_openfst" == "xyes"], [AC_MSG_NOTICE([openfst will be used])], [AC_MSG_NOTICE([openfst will not be used])])
AS_IF([test "x$have_openfst" == "xyes"], [AC_DEFINE([WITH_OPENFST], [1], [Used in source files])])
AM_CONDITIONAL(WITH_OPENFST, test x$have_openfst = xyes)

AC_CHECK_HEADERS([lm/model.hh boost/iostreams/filter/gzip.hpp], [] ,[AC_MSG_ERROR([some headers were not found. set CPPFLAGS to header location])] ,[])

LIBS+=" -ldl -lfst"
AC_CHECK_HEADERS([fst/fstlib.h], [] ,[AC_MSG_ERROR([some headers were not found. set CPPFLAGS to header location])] ,[])
AC_MSG_CHECKING([for -lfst])
AC_LINK_IFELSE(
  [AC_LANG_PROGRAM([#include <fst/fstlib.h>],[])],
  [AC_MSG_RESULT([yes])],
  [AC_MSG_RESULT([no])]
  [AC_MSG_ERROR([openfst is not found. set LDFLAGS to library location])]
  )

LIBS+=" -lboost_iostreams"
AC_ARG_WITH(snappy,
        [AC_HELP_STRING([--with-snappy=DIR], [snappy in DIR])],
        [snappy_dir="${withval}"], [snappy_dir=''])
AC_ARG_ENABLE(snappy,
        [AC_HELP_STRING([--enable-snappy], [enable snappy])],
        [ac_enable_snappy=yes], [ac_enable_snappy=no])

if test "x$snappy_dir" != "x"; then
  SNAPPY_CPPFLAGS="-I${snappy_dir}/include"
  SNAPPY_LDFLAGS="-L${snappy_dir}/lib -lsnappy"
else
  SNAPPY_CPPFLAGS=""
  SNAPPY_LDFLAGS="-lsnappy"
fi

if test "x$ac_enable_snappy" = "xyes"; then
   AC_MSG_CHECKING(for snappy)
   
   save_CXXFLAGS="$CXXFLAGS"
   save_LDFLAGS="$LDFLAGS"
   CXXFLAGS="$CXXFLAGS $SNAPPY_CPPFLAGS"
   LDFLAGS="$LDFLAGS $SNAPPY_LDFLAGS"

   AC_TRY_LINK([], [], [ac_have_libsnappy=yes], [ac_have_libsnappy=no])
   AC_MSG_RESULT($ac_have_libsnappy)
   AC_CHECK_HEADERS([snappy.h])

   if test "x$ac_have_libsnappy" = "xno" -o "x$ac_cv_header_snappy_h" = "xno"; then
     SNAPPY_CPPFLAGS=""
     SNAPPY_LDFLAGS=""
   else
     AC_DEFINE(HAVE_SNAPPY, 1, [Define if you have snappy])
   fi

  CXXFLAGS="$save_CXXFLAGS"
  LDFLAGS="$save_LDFLAGS"
else
  SNAPPY_CPPFLAGS=""
  SNAPPY_LDFLAGS=""
fi
AC_SUBST(SNAPPY_CPPFLAGS)
AC_SUBST(SNAPPY_LDFLAGS)
AM_CONDITIONAL([WITH_SNAPPY], [test "x$ac_have_libsnappy" = "xyes" -a "x$ac_cv_header_snappy_h" = "xyes"])

AC_MSG_CHECKING([for -lkenlm])
LIBS+=" -lkenlm -lboost_thread"
AC_LINK_IFELSE(
  [AC_LANG_PROGRAM([#include <lm/model.hh>],
    [lm::ngram::Vocabulary()])],
  [AC_MSG_RESULT([yes])],
  [AC_MSG_RESULT([no])]
  [AC_MSG_ERROR([kenlm is not found. set LDFLAGS to library location])]
)


AC_MSG_CHECKING([for -lboost_iostreams])
AC_LINK_IFELSE(
  [AC_LANG_PROGRAM([#include <boost/iostreams/filter/gzip.hpp>],
    [boost::iostreams::gzip_decompressor();])],
  [AC_MSG_RESULT([yes])],
  [AC_MSG_RESULT([no])]
  [AC_MSG_ERROR([boost_iostreams is not found. set LDFLAGS to library location])]
)

LIBS="$save_LIBS"

AC_CONFIG_FILES([Makefile
                 codec/Makefile
                 template/Makefile])
AC_OUTPUT
