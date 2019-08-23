#include "template.hh"
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <limits>
#include <gflags/gflags.h>
#include <libconfig.h++>

using std::cerr;
using std::endl;

DEFINE_bool(hello, false, "say hello?");
DEFINE_string(config, "", "config file to read from");

int main(int argc, char ** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // flags
  if (FLAGS_hello) {
    std::cerr << "hello" << std::endl;
  } else {
    std::cerr << "goodbye" << std::endl;
  }

  if (!FLAGS_config.empty()) {
    // config files
    libconfig::Config cfg;
    try {
      cfg.readFile(FLAGS_config.c_str());
    } catch(const libconfig::FileIOException &fioex) {
    }
    const libconfig::Setting & sel = cfg.getRoot();
  }
}
