#ifndef MISC_IO_H_
#define MISC_IO_H_

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

#include "global.hh"

using std::endl;

namespace misc {

long getStreamSize(std::ifstream& ifs);
long getFileSize(const std::string &fn);
bool path_exists(const std::string &path);
bool file_exists(const std::string &filename);

std::string expand_special_input_filenames(const std::string& fn);
std::string expand_special_output_filenames(const std::string& fn);
std::string get_temp_fn();
std::string make_file(const std::string &content);
std::string decide_path(const std::string &config_dir, const std::string &fn);
std::vector<std::string> readCorpus(const std::string &fn, size_t max_lines = 0);

class OFileStream {
public:
    OFileStream(const std::string &fn);
    virtual ~OFileStream();
    boost::iostreams::filtering_ostream& get();
    std::ofstream& getOfstream();
private:
    std::ofstream ofstr;
    boost::iostreams::filtering_ostream fofstr;
};

class IFileStream {
public:
    IFileStream(const std::string &fn);
    virtual ~IFileStream();
    boost::iostreams::filtering_istream& get();
    std::ifstream& getIfstream();
    size_t getFilesize() {
      return filesize;
    }
private:
    std::ifstream ifstr;
    boost::iostreams::filtering_istream fifstr;
    size_t filesize;
};
}
#endif /* MISC_IO_H_ */
