#include "misc_time.hh"

namespace misc {

timeval start_time;

time_t compile_time(char const *time) { 
  char s_month[5];
  int month, day, year;
  int hour, minute, second;
  struct tm t = {0};
  static const char month_names[] = "JanFebMarAprMayJunJulAugSepOctNovDec";

  sscanf(time, "%s %d %d %d:%d:%d", s_month, &day, &year, &hour, &minute, &second);

  month = (strstr(month_names, s_month)-month_names)/3;

  t.tm_mon = month;
  t.tm_mday = day;
  t.tm_year = year - 1900;
  t.tm_isdst = -1;
  t.tm_hour = hour;
  t.tm_min = minute;
  t.tm_sec = second;

  return mktime(&t);
}

long get_seed() {
  struct timeval time;
  gettimeofday(&time, NULL);
  long seed = (time.tv_sec * 1000 + time.tv_usec / 1000);
  LOG(INFO) << "generated seed " << seed << std::endl;
  return seed;
}

long getTimeSecs() {
  timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec;
}

long getTimeMillis() {
  timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec * 1000 + time.tv_usec / 1000;
}

long getTimeMicros() {
  timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec * 1000 * 1000 + time.tv_usec;
}

long getTimeNanos() {
  timespec ts;

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts.tv_sec = mts.tv_sec;
  ts.tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_REALTIME, &ts);
#endif
  return ts.tv_sec * 1000 * 1000 * 1000 + ts.tv_nsec;
}

void start() { gettimeofday(&start_time, NULL); }

double seconds() {
  struct timeval now;
  gettimeofday(&now, NULL);
  double seconds = ((now.tv_sec - start_time.tv_sec) * 1000000 +
                    (now.tv_usec - start_time.tv_usec)) /
                   1000000.0;
  return seconds;
}

const std::string now() {
  char buf[80];
  sprintf(buf, "[%11.5f]", seconds());
  // strftime(buf,sizeof(buf),"%Y-%m-%d.%X", &tstruct);
  return buf;
}


ProcessStopWatch::ProcessStopWatch() {
  reset();
}

void ProcessStopWatch::reset() {
  user_start_ = boost::chrono::process_user_cpu_clock::now();
  wall_start_ = boost::chrono::system_clock::now();
}

void ProcessStopWatch::store() {
  user_end_ = boost::chrono::process_user_cpu_clock::now();
  wall_end_ = boost::chrono::system_clock::now();
}


boost::chrono::milliseconds ProcessStopWatch::user_millis() const {
  return boost::chrono::duration_cast<boost::chrono::milliseconds>(
      user_end_ - user_start_);
}

boost::chrono::milliseconds ProcessStopWatch::wall_millis() const {
  return boost::chrono::duration_cast<boost::chrono::milliseconds>(
      wall_end_ - wall_start_);
}

}
