#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "json.hpp"

using json = nlohmann::json;

// This kernel creates bank conflicts ranging from 0 to 31 per warp for loads.
// Parameter "conflicts" controls the number of conflicts to be simulated.
// If conflicts == 0, then this is same as microben1, which just creates wavefronts without conflicts.
__global__ void microben(int *arr, long long int *ticks, long long int *start, long long int *end, int conflicts)
{
  __shared__ int s[8192];
  int addr;
  int var;
  int th = threadIdx.x;
  long long int t1, t2;
  //s[th] = arr[th];
  //__syncthreads();
  addr = th % 32;
  if(conflicts != 0) {
    addr = addr % (conflicts + 1);
    addr = addr * 32;
  }
  __syncthreads();
  //addr = addr*32;
  //addr = th * 32;
  t1 = clock64();
  var = s[addr];
  var = var + 1;
  t2 = clock64();
  start[th] = t1;
  end[th] = t2;
  ticks[th] = t2-t1;
  arr[th] = var;
}
//static void write_csv(std::string filename, )
int main(int argc, char** argv)
{
    const int threads = atoi(argv[1]);
    const int conflicts = atoi(argv[2]);
    //std::cout << "Threads Arg: " << threads << std::endl;
    //std::cout << "Conflicts Arg: " << conflicts << std::endl;

    const int n = 8192;
    std::vector <long long int> h_start, h_end, h_time;
    h_start = std::vector<long long int>(threads, std::numeric_limits<long long int>::max());
    h_end = std::vector<long long int>(threads, std::numeric_limits<long long int>::min());
    h_time = std::vector<long long int>(threads, std::numeric_limits<long long int>::min());
  
    int h_A[n], h_B[n];
    int *d_A;
    long long int *d_time, *d_start, *d_end;
    
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = 0;
    }
    
    cudaMalloc(&d_A, n * sizeof(int)); 
    cudaMalloc(&d_time, threads * sizeof(long long int));
    cudaMalloc(&d_start, threads * sizeof(long long int));
    cudaMalloc(&d_end, threads * sizeof(long long int));
    
    cudaMemcpy(d_A, h_A, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_time, h_time.data(), threads*sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, h_start.data(), threads*sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, h_end.data(), threads*sizeof(long long int), cudaMemcpyHostToDevice);

    //microben1<<<1,threads>>>(d_A, d_time, d_start, d_end, conflicts);
    microben<<<1,threads>>>(d_A, d_time, d_start, d_end, conflicts);

    cudaMemcpy(h_B, d_A, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_time.data(), d_time, threads*sizeof(long long int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_start.data(), d_start, threads*sizeof(long long int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end.data(), d_end, threads*sizeof(long long int), cudaMemcpyDeviceToHost);

    long long int time = *std::max_element(h_end.cbegin(), h_end.cend()) - *std::min_element(h_start.cbegin(), h_start.cend());
    nlohmann::json obj;
    const int warps = ceil(float(threads)/32.0);
    obj["warps"] = warps;
    obj["Conflicts"] = conflicts;
    // obj["wid"] = nlohmann::json::array();
    // obj["thid"] = nlohmann::json::array();
    // obj["tic"] = nlohmann::json::array();
    // obj["toc"] = nlohmann::json::array();
    // obj["cycles"] = nlohmann::json::array();
    obj["T"] = time;
    //long long int time = *std::max_element(h_end.cbegin(), h_end.cend()) - *std::min_element(h_start.cbegin(), h_start.cend());

    long long int basevalue = *std::min_element(h_start.cbegin(), h_start.cend()); 
    // for(int i=0; i < threads; i++){
    //     std::cout << "Thread " << i << " Start: " << (h_start[i]-h_start[threads-1]) << " End:" << (h_end[i]-h_start[threads-1]) << " Clock_Cycles: " <<  h_time[i] << std::endl;
    //     // obj["wid"].push_back(i/32);
    //     // obj["thid"].push_back(i);
    //     // obj["tic"].push_back(h_start[i]-basevalue);
    //     // obj["toc"].push_back(h_end[i]-basevalue);
    //     // obj["cycles"].push_back(h_time[i]);
    // }
    //std::cout << obj << std::endl;
    //std::ofstream fp("load_model.csv");
    //fp << "Threads" << "," << "Conflicts" << "," << "Time" << std::endl;
    
    
    //std::cout << obj << std::endl;
    //std::cout << warps << "," << conflicts << "," << time << std::endl;
    std::cout << time << std::endl;

    //fp.close();

    cudaFree(d_A);
    cudaFree(d_time);
    cudaFree(d_start);
    cudaFree(d_end);

    return 0;
}