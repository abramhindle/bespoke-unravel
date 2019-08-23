#ifndef TEMPLATE_H_
#define TEMPLATE_H_

#include <boost/iostreams/filtering_stream.hpp>
#include <map>
#include <unordered_map>
#include <ostream>
#include <string>
#include <vector>

class Template {
public:
    Template();
    ~Template();
    std::string getTemplate() const;
};
#endif /* TEMPLATE_H_ */
