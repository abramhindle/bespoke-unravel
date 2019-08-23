#ifndef MISC_PRETTYPRINT_H_
#define MISC_PRETTYPRINT_H_

#include "misc_time.hh"
#include <glog/logging.h>
#include <iomanip>  
#include <iostream>

namespace misc {

extern double last_seconds;
extern std::ostream null_stream;

std::string intToHumanStr(size_t num);
std::string intToStrZeroFill(size_t num, size_t length);
std::string blanks(int n);
std::ostream& update(double interval, bool force);
std::ostream& clear_line();
std::ostream& log();
std::ostream& clear_line(std::ostream& ostr);
std::ostream& log(std::ostream& ostr);
void printTable(std::vector<std::string const*> const& lines);
void printTable(std::string const* line1, std::string const* line2);

}

#endif /* MISC_PRETTYPRINT_H_ */
