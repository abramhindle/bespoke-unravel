#include <gtest/gtest.h>
#include "template.hh"

TEST(TemplateTestSuite, TemplateTestGet) {
  Template t;

  EXPECT_EQ("yes",t.getTemplate()) << "works...";
}
