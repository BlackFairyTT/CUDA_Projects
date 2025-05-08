/* 
________________________________________________

CUDA Vector Addition
________________________________________________

Implement program that performs element-wise addition of two vectors containing 
32-bit floating point numbers on a GPU. The program should take two input vectors of 
equal length and produce a single output vector containing their sum. 

Test Link : https://leetgpu.com/playground 
[Copy and Paste the code into the LeetGPU playground.cu window and click "Run" to execute]
[he timing results from LeetGPU can be misleading, as the slower GPU execution times are 
likely due to overhead introduced by the sandboxed environment rather than actual kernel performance.]

Output : 
    Running NVIDIA GTX TITAN X in FUNCTIONAL mode...
    Compiling...
    Executing...
    GPU Naive Kernel time: 28.2395 ms
    GPU Optimized Kernel time: 92.4052 ms
    CPU time: 3.81672 ms
    Results match!
    Exit status: 0
*/

#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

/* 

🔍 Difference between runNaiveKernel() and runOptimizedKernel(
_________________________________________________________________________________________________
Feature	               | runNaiveKernel()	        runOptimizedKernel()
_________________________________________________________________________________________________
Kernel	               | vector_add	                | vector_add_Opt
_________________________________________________________________________________________________
Threads	               | Each thread does 1 element	| Each thread does multiple elements
_________________________________________________________________________________________________
Uses grid-stride loop  | ❌ No	                   | ✅ Yes
_________________________________________________________________________________________________
Resource utilization.  | ❌ Poor for large N	       | ✅ Better scalability
_________________________________________________________________________________________________
Occupancy	           | May launch too many blocks	| Can launch fewer blocks & still saturate GPU
_________________________________________________________________________________________________

🧠 Rule of Thumb

    For small arrays (e.g. N < 10,000): naive is simpler and fine.

    For medium to large arrays (e.g. N ≥ 100,000): use optimized with grid-stride loop.

*/
template <typename T>
__global__ void vector_add(const T* A, const T* B, T* C, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N)
        C[id] = A[id] + B[id];
}

/* 
Sum of arbitrary long vectors :
When you have a large array of data that exceeds the number of threads, 
you can divide the workload using grid-stride loops. This allows each thread to handle 
multiple elements instead of just one, which improves efficiency, especially for large arrays.*/
template <typename T>
__global__ void vector_add_Opt(const T* __restrict__ A,
                               const T* __restrict__ B,
                               T* C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Shift by the total number of thread in a grid
    // gridDim.x * blockDim.x: Total number of threads in the entire grid
    int stride = gridDim.x * blockDim.x; 

#pragma unroll
    for (int i = id; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

template <typename KernelFunc>
float measure_gpu_time(KernelFunc kernel) {
    cudaEvent_t start, stop;
    float ms = 0;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    kernel();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

template <typename CpuFunc>
double measure_cpu_time(CpuFunc func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename T>
class VectorAdder {
public:
    VectorAdder(int N) : N(N), size(N * sizeof(T)) {
        h_A = new T[N];
        h_B = new T[N];
        h_C = new T[N];
        h_C_CPU = new T[N];

        for (int i = 0; i < N; ++i) {
            h_A[i] = static_cast<T>(i * 1.5);
            h_B[i] = static_cast<T>((i + 1) * 2.5);
        }

        CHECK_CUDA(cudaMalloc(&d_A, size));
        CHECK_CUDA(cudaMalloc(&d_B, size));
        CHECK_CUDA(cudaMalloc(&d_C, size));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    }

    ~VectorAdder() {
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_C_CPU;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void runGpu() {
        float timeNaive = measure_gpu_time([&]() { runNaiveKernel(); });
        std::cout << "GPU Naive Kernel time: " << timeNaive << " ms\n";

        // Reset Device Output Buffer
        CHECK_CUDA(cudaMemset(d_C, 0, size));

        float timeOpt = measure_gpu_time([&]() { runOptimizedKernel(); });
        std::cout << "GPU Optimized Kernel time: " << timeOpt << " ms\n";

        CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    }

    void runCpu() {
        double timeCPU = measure_cpu_time([&]() {
            for (int i = 0; i < N; ++i)
                h_C_CPU[i] = h_A[i] + h_B[i];
        });
        std::cout << "CPU time: " << timeCPU << " ms\n";
    }

    void validate() const {
        int errors = 0;
        for (int i = 0; i < N; ++i) {
            if (fabs(h_C[i] - h_C_CPU[i]) > 1e-5f) {
                if (++errors <= 10)
                    std::cout << "Mismatch at " << i << ": GPU " << h_C[i]
                              << " CPU " << h_C_CPU[i] << "\n";
            }
        }

        if (errors == 0)
            std::cout << "Results match!\n";
        else
            std::cout << "Total mismatches: " << errors << "\n";
    }

private:
    int N;
    size_t size;
    T *h_A, *h_B, *h_C, *h_C_CPU;
    T *d_A, *d_B, *d_C;

    /* | Kernel    | Launch Configuration                                                                 | Comments                           |
        | --------- | ----------------------------------------------------------------------------------- | ---------------------------------- |
        | Naive     | `blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock`                       | 1 element per thread               |
        | Optimized | `blocksPerGrid = ((N / elementsPerThread) + threadsPerBlock - 1) / threadsPerBlock` | Fewer threads, each processes more |
    */
    void runNaiveKernel() {
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void runOptimizedKernel() { // Need to test in actual Machine
        int elementsPerThread = 4;
        int totalThreads = (N + elementsPerThread - 1) / elementsPerThread;
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

        vector_add_Opt<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
};

int main() {
    const int N = 1024 * 1024;

    // Change float to double if needed
    VectorAdder<float> adder(N);

    adder.runGpu();
    adder.runCpu();
    adder.validate();

    return 0;
}
