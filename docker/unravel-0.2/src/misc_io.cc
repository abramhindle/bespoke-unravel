#include "misc_io.hh"

#include <glog/logging.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <codec/lz4.hpp>
#include <codec/quicklz.hpp>
#include <codec/fastlz.hpp>


using std::endl;

namespace misc {

std::string expand_special_output_filenames(const std::string& fn) {
  if (fn == "-") {
    return "/dev/stdout";
  } else if (fn == "0") {
    return "/dev/null";
  } else {
    return fn;
  }
}

std::string expand_special_input_filenames(const std::string& fn) {
  if (fn == "-") {
    return "/dev/stdin";
  } else if (fn == "0") {
    return "/dev/null";
  } else {
    return fn;
  }
}

bool path_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

long getStreamSize(std::ifstream& ifs) {
  ifs.seekg(0, std::ifstream::end);
  size_t total_bytes = ifs.tellg();
  ifs.seekg(0, std::ifstream::beg);
  return total_bytes;
}

long getFileSize(const std::string &fn) {
  struct stat stat_buf;
  int rc = stat(fn.c_str(), &stat_buf);
  return rc == 0 ? stat_buf.st_size : -1;
}

bool file_exists(const std::string &filename) {
  std::ifstream ifile(filename);
  return ifile.is_open();
}

std::string get_temp_fn() {
  char filename[] = "/tmp/decipher.tmp.XXXXXX";
  int fd = mkstemp(filename);
  CHECK(fd!=-1);
  close(fd);
  VLOG(1) << "created temp file '" << filename << "'";
  return filename;
}

std::string make_file(const std::string &content) {
  std::string fn = get_temp_fn();
  VLOG(1) << "filling file '" << fn << "' with " << content.size() << " bytes of data";
  std::ofstream ostr(fn);
  ostr << content;
  return fn;
}

std::string decide_path(const std::string &config_dir, const std::string &fn) {
  std::string filename;

  if (fn[0] != '/') {
    filename = config_dir + "/" + fn;
  } else {
    filename = fn;
  }

  CHECK(file_exists(filename)) << "file does not exist: " << filename;

  return filename;
}

OFileStream::OFileStream(const std::string &fn) {
  std::string filename = expand_special_output_filenames(fn);
  ofstr.open(filename.c_str(), std::ios_base::out | std::ios_base::binary);
  CHECK(ofstr.is_open()) << "Error: opening file \"" << filename << "\"";

  std::string compressor_info = "plaintext";

  if (boost::algorithm::ends_with(filename, ".lz4")) {
    compressor_info = "lz4";
    fofstr.push(codec::lz4_compressor());
  } else if (boost::algorithm::ends_with(filename, ".qlz")) {
    compressor_info = "qlz";
    fofstr.push(codec::quicklz_compressor());
  } else if (boost::algorithm::ends_with(filename, ".flz")) {
    compressor_info = "flz";
    fofstr.push(codec::fastlz_compressor());
  } else if (boost::algorithm::ends_with(filename, ".gz")) {
    compressor_info = "gz";
    fofstr.push(boost::iostreams::gzip_compressor());
  }
  fofstr.push(ofstr);

  LOG(INFO) << "opening file '" << filename << "' for writing, compressor=" <<
    compressor_info;
}

std::ofstream& OFileStream::getOfstream() { return ofstr; }
boost::iostreams::filtering_ostream& OFileStream::get() { return fofstr; }
OFileStream::~OFileStream() {}

IFileStream::IFileStream(const std::string &fn) {
  std::string filename = expand_special_input_filenames(fn);
  ifstr.open(filename.c_str(), std::ios_base::in | std::ios_base::binary);
  filesize = misc::getStreamSize(ifstr);

  CHECK(ifstr.is_open()) << "Error: opening file \"" << filename << "\"";

  std::string decompressor_info = "plaintext";

  if (boost::algorithm::ends_with(filename, ".lz4")) {
    decompressor_info = "lz4";
    fifstr.push(codec::lz4_decompressor());
  } else if (boost::algorithm::ends_with(filename, ".qlz")) {
    decompressor_info = "qlz";
    fifstr.push(codec::quicklz_decompressor());
  } else if (boost::algorithm::ends_with(filename, ".flz")) {
    decompressor_info = "flz";
    fifstr.push(codec::fastlz_decompressor());
  } else if (boost::algorithm::ends_with(filename, ".gz")) {
    decompressor_info = "gz";
    fifstr.push(boost::iostreams::gzip_decompressor());
  }
  fifstr.push(ifstr);

  LOG(INFO) << "opening file '" << filename << "' for reading, size=" <<
    filesize << ", decompressor=" << decompressor_info;
}

std::ifstream& IFileStream::getIfstream() { return ifstr; }
boost::iostreams::filtering_istream& IFileStream::get() { return fifstr; }
IFileStream::~IFileStream() {}

std::vector<std::string> readCorpus(const std::string &fn, size_t max_lines) {
  std::vector<std::string> result;
  LOG(INFO) << "read corpus from '" << fn << "' max_lines=" << max_lines;
  IFileStream ifs(fn);

  for (std::string str; std::getline(ifs.get(), str);) {
    boost::algorithm::trim(str);
    result.push_back(str);
    if (max_lines > 0 && result.size() > max_lines) break;
  }

  LOG(INFO) << "done reading " << result.size() << " lines from '" << fn << "'";
  return result;
}
}
