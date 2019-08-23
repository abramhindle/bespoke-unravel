#include "template.hh"
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <limits>

using std::cerr;
using std::endl;

Template::Template() {
}

Template::~Template() {
}

/** returns the string representation of the given int representation */
std::string Template::getTemplate() const {
  return "yes";
}
