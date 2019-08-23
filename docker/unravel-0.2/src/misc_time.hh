#ifndef MISC_TIME_H_
#define MISC_TIME_H_

#include <glog/logging.h>
#include <sys/time.h>
#include <time.h>
#include <ctime>
#include <boost/chrono.hpp>
#include <boost/chrono/duration.hpp>

#ifdef __MACH__
#include<mach/mach.h>
#include<mach/clock.h>
#endif

namespace misc {

extern timeval start_time;

time_t compile_time(char const *time = __DATE__ " " __TIME__);
long get_seed();
long getTimeSecs();
long getTimeMillis();
long getTimeMicros();
long getTimeNanos();
double seconds();
const std::string now();

class ProcessStopWatch {
 public:
  ProcessStopWatch();
  void reset();
  void store();
  boost::chrono::milliseconds user_millis() const;
  boost::chrono::milliseconds wall_millis() const;

 private:
  boost::chrono::time_point<boost::chrono::process_user_cpu_clock> user_start_;
  boost::chrono::time_point<boost::chrono::system_clock> wall_start_;
  boost::chrono::time_point<boost::chrono::process_user_cpu_clock> user_end_;
  boost::chrono::time_point<boost::chrono::system_clock> wall_end_;
};

}

#endif /* MISC_TIME_H_ */
