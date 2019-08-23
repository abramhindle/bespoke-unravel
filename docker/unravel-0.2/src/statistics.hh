#ifndef STATISTICS_H__
#define STATISTICS_H__

// todo MAC
//#include <sys/sysinfo.h>
#include <glog/logging.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <ctime>
#include "misc.hh"
#include "misc_prettyprint.hh"

using std::cerr;

static struct timeval STATISTICS_start_time;
static struct timeval STATISTICS_last_dump_time;

inline void process_mem_usage(double& vm_usage, double& resident_set) {
    using std::ios_base;
    using std::ifstream;
    using std::string;

    vm_usage     = 0.0;
    resident_set = 0.0;

    // 'file' stat seems to give the most reliable results
    ifstream stat_stream("/proc/self/stat",ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> nice
                >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage     = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

inline void init_statistics() {
    gettimeofday(&STATISTICS_start_time, NULL);
}

inline void dump_statistics() {
    gettimeofday(&STATISTICS_last_dump_time, NULL);
    //struct sysinfo info;
    struct rusage usage;
    //sysinfo(&info);
    getrusage(RUSAGE_SELF,&usage);
    struct timeval cur_time;
    gettimeofday(&cur_time, NULL);
    double run_time_s = (cur_time.tv_sec-STATISTICS_start_time.tv_sec) + (cur_time.tv_usec-STATISTICS_start_time.tv_usec) / (1000*1000) ;
    double user_time_s = (usage.ru_utime.tv_sec + (1.0*usage.ru_utime.tv_usec/1000)/1000.0);
    double sys_time_s = (usage.ru_stime.tv_sec + (1.0*usage.ru_stime.tv_usec/1000)/1000.0);
    double vm_usage;
    double resident_set;
    clock_t cpu_usage = clock();
    double cpu_usage_s = double(cpu_usage/CLOCKS_PER_SEC);
    process_mem_usage(vm_usage, resident_set);
    LOG(INFO) << "run/user/sys/cpu time "
              << misc::intToHumanStr(run_time_s) << "s"
              << "/"   << misc::intToHumanStr(user_time_s) << "s"
              << "/"    << misc::intToHumanStr(sys_time_s) << "s"
              << "/"    << misc::intToHumanStr(cpu_usage_s) << "s";
    LOG(INFO) << "parallelism " << user_time_s/run_time_s;
    LOG(INFO) << "VM "          << misc::intToHumanStr(vm_usage) << "B\t"
              << "RSS "         << misc::intToHumanStr(resident_set) << "B";
    //LOG(INFO) << "SYS_MEM " << info.freeram/(1024*1024*1024)*info.mem_unit << "GB / " << info.totalram/(1024*1024*1024)*info.mem_unit << "GB" << std::endl;
}

inline void dump_every(size_t seconds) {
    struct timeval cur_time;
    gettimeofday(&cur_time, NULL);
    double time_since_last_dump_s = (cur_time.tv_sec-STATISTICS_last_dump_time.tv_sec) + (cur_time.tv_usec-STATISTICS_last_dump_time.tv_usec) / (1000*1000) ;
    if (time_since_last_dump_s > seconds) dump_statistics();
}

#endif
