#include "misc_prettyprint.hh"

namespace misc {

double last_seconds = 0;
std::ostream null_stream(0);

std::string intToHumanStr(size_t num) {
  double d_num = static_cast<double>(num);
  int exponent = 0;

  for (size_t i = 0; i < 5; ++i) {
    if (d_num > 1000.0) {
      ++exponent;
      d_num /= 1000.0;
    } else {
      break;
    }
  }
  std::vector<std::string> exts = {"", "k", "M", "G", "T", "P", "E", "Z", "Y"};
  std::ostringstream iter_sstream;
  iter_sstream << std::fixed << std::setprecision(2) << std::setfill('0')
               << std::setw(4) << d_num << exts[exponent];
  return iter_sstream.str();
}

std::string intToStrZeroFill(size_t num, size_t length) {
  std::ostringstream iter_sstream;
  iter_sstream << std::setw(length) << std::setfill('0') << num;
  return iter_sstream.str();
}


std::string blanks(int n) {
  std::string blanks = "";
  for (int i = 0; i < n; ++i) {
    blanks += " ";
  }
  return blanks;
}

std::ostream& update(double interval, bool force) {
  if (seconds() - last_seconds > interval || force) {
    last_seconds = seconds();
    return log();
  } else {
    return null_stream;
  }
}

std::ostream& clear_line() { return std::cerr << "\33[K"; }

std::ostream& log() { return std::cerr << now() << " "; }

std::ostream& clear_line(std::ostream& ostr) { return ostr << "\33[K"; }

std::ostream& log(std::ostream& ostr) { return ostr << now() << " "; }

void printTable(std::vector<std::string const*> const& lines) {
  size_t n_lines = lines.size();
  std::vector<std::vector<std::string> > columnTokens(n_lines);
  unsigned int maxColumns = 0;
  for (size_t i = 0; i < n_lines; ++i) {
    std::stringstream stream(*(lines[i]));
    unsigned int count = 0;
    while (!stream.eof()) {
      std::string token;
      std::getline(stream, token, ' ');
      if (!token.empty()) {
        columnTokens[i].push_back(token);
        ++count;
      }
    }
    if (count > maxColumns) {
      maxColumns = count;
    }
  }

  for (unsigned int j = 0; j < maxColumns; ++j) {
    size_t maxLength = 0;
    for (size_t i = 0; i < n_lines; ++i) {
      if (columnTokens[i].size() > j) {
        size_t length = columnTokens[i][j].length();
        if (length > maxLength) {
          maxLength = length;
        }
      }
    }
    for (size_t i = 0; i < n_lines; ++i) {
      if (columnTokens[i].size() > j) {
        columnTokens[i][j] =
            columnTokens[i][j] +
            blanks(maxLength - columnTokens[i][j].length() + 2);
      }
    }
  }

  for (size_t i = 0; i < n_lines; ++i) {
    for (size_t j = 0; j < maxColumns; ++j) {
      if (columnTokens[i].size() > j) {
        std::cout << columnTokens[i][j];
      }
    }
    std::cout << std::endl;
  }
}

void printTable(std::string const* line1, std::string const* line2) {
  std::vector<std::string const*> lines(2);
  lines[0] = line1;
  lines[1] = line2;
  printTable(lines);
}

}
