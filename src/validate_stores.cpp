//#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <sstream>
#include "rapidcsv.h"
#include <regex>
using namespace std;


struct NCUcounters
{
    //uint64_t inst_loads;
    uint64_t inst_stores;
    //uint64_t conflicts_loads;
    uint64_t conflicts_stores;
    //uint64_t wavefronts_loads;
    uint64_t wavefronts_stores;
};

static uint64_t getNCUMetric(const std::map<std::string, std::string> metrics,
                              const std::string& metric) {
  auto s = metrics.at(metric);
  s.erase(std::remove(s.begin(), s.end(), ','), s.end());
  return std::stoull(s);
}

static uint64_t parsecycles(std::stringstream& ss)
{
    std::regex re_prof(R"(==PROF== Connected.*)");
    std::string line;
    bool got_profile = false;
    while (std::getline(ss, line)) 
    {
        //std::cout << line << std::endl;   
        if (std::regex_search(line, re_prof)) 
        {
            got_profile = true;
            break;
        }
    }
    //std::getline(ss, line);
    //std::cout << line << std::endl;
    std::getline(ss, line, '\n');
    uint64_t time = std::stoull(line);
    return(time);
}

static NCUcounters parseNCU(std::stringstream& ss)
{
    //std::cout << ss.str() << endl;
    std::string line;
    NCUcounters counters;
    std::regex prof(R"(==PROF== Disconnected.*)");
    bool got_profile = false;
    std::stringstream csv;
    while(std::getline(ss, line))
    {
        //std::cout << line << std::endl;
        if (std::regex_search(line, prof))
        {
            got_profile = true;
            break;
        }
    }
    //std::getline(ss, line);
    //std::cout << line << std::endl;
    // Get all rows
    while(std::getline(ss, line)) {
        //cout << line << std::endl;
        csv << line << std::endl;
    }

    //std::cout << csv.str() << std::endl;
    rapidcsv::Document doc(csv, rapidcsv::LabelParams(),
                         rapidcsv::SeparatorParams(),
                         rapidcsv::ConverterParams(),
                         rapidcsv::LineReaderParams(false, '#', true));
    std::map<std::string, std::string> metrics;
    for (int i = 0; i < doc.GetRowCount(); ++i) {
        auto metric = doc.GetCell<std::string>("Metric Name", i);
        auto value = doc.GetCell<std::string>("Metric Value", i);
        metrics[metric] = value;
        //std::cout << "Metric Name: " << metric << "Metric Value: " << value << std::endl;
    }
    counters.wavefronts_stores = getNCUMetric(metrics, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum");
    counters.conflicts_stores = getNCUMetric(metrics, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum");
    counters.inst_stores = getNCUMetric(metrics, "sass__inst_executed_shared_stores");
    return counters;

    //cout << "Load Wavefronts:" << wavefront << endl;
    //cout << "Row count" << doc.GetRowCount() << endl;

}

int main()
{ 
    std::string str, str_ncu, str_exe;
    const int max_threads = 1024;

    // Loading model for loads
    rapidcsv::Document model("store_model.csv");
    std::map<std::tuple<int, int>, int> tup;
    for (int i = 0; i < model.GetRowCount(); ++i) 
    {
        auto n = model.GetCell<int>("Warps", i);
        auto e = model.GetCell<int>("Conflicts", i);
        auto t = model.GetCell<int>("Time", i);
        tup[std::make_tuple(n, e)] = t;
    }

    //Creating a csv to output measured and model timing values
    std::ofstream timing("store_timing.csv");
    timing << "Warps" << "," << "Conflicts" << "," << "Measured" << "," << "Model" << std::endl; 

    for (int threads = 32; threads <= max_threads; threads=threads+32)
    {
        //for (int conflicts = 0; conflicts <= (threads % 32); conflicts++)
        for (int conflicts = 0; conflicts <= 31; conflicts++)
        {
            FILE *fp;
            char buf[1024];
            std::stringstream ss, ss_dup;
            str_ncu = "ncu --csv --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,sass__inst_executed_shared_stores";
            str_exe = " microstores.exe ";
            str = str_ncu + str_exe + std::to_string(threads) + " " + std::to_string(conflicts) + " > ncu_buff.txt";
            const char *command = str.c_str();
            //std::cout << "Command is " << command << endl;
            system(command);
            fp = std::fopen("ncu_buff.txt", "r");
            if(fp == NULL)
            {
                cout << "Error opening file" << std::endl;
            }
            
            while(fgets(buf, 1024, fp))
            {
                ss << buf;
                ss_dup << buf;
            }
            fclose(fp);
            //std::cout << ss.str() << std::endl;

            NCUcounters counters = parseNCU(ss);
            uint64_t measured_time = parsecycles(ss_dup);
            int ConflictsperWarp = counters.conflicts_stores/counters.inst_stores;
            uint64_t model_time = tup.at(std::make_tuple(counters.inst_stores, ConflictsperWarp));
            //cout << "Measured Time: " << measured_time << " Model Time: " << model_time << endl;
            //const int warps = ceil(float(threads)/32.0);
            timing << counters.inst_stores << "," << ConflictsperWarp << "," << measured_time << "," << model_time << std::endl;

        }
    }
    timing.close();

    return 0;
}