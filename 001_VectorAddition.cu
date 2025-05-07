/* 
________________________________________________

CUDA Vector Addition
________________________________________________

Implement program that performs element-wise addition of two vectors containing 
32-bit floating point numbers on a GPU. The program should take two input vectors of 
equal length and produce a single output vector containing their sum. 

Test Link : https://leetgpu.com/playground [Copy and Paste the code into the LeetGPU playground.cu window and click "Run" to execute]

Output : 
    Running NVIDIA GTX TITAN X in FUNCTIONAL mode...
    Compiling...
    Executing...
    GPU Naive Kernel time: 0.003057 ms
    GPU Optimized Kernel time: 0.002831 ms
    CPU time: 3.90582 ms
    Results match!
    Exit status: 0
*/

#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// CUDA Vector Addition 
__global__ void vector_add(const float* A, const float* B, float* C, int N) { // Naive kernel

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < N)
        C[id] = A[id] + B[id];
}

// CUDA Vector Addition using grid stride and loop unroll
__global__ void vector_add_Opt(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* C, int N) { // Optimized kernel

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

#pragma unroll
    for (int i = id; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// To call vector_add
void solveN(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    //cudaDeviceSynchronize();
}

// To call vector_add_Opt
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_Opt<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    //cudaDeviceSynchronize();
}

// C++ Vector Addition 
void solveC(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1024 * 1024; 
    size_t size = N * sizeof(float);

    // Host memory (raw arrays)
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];
    float* h_C_CPU = new float[N];

    // Initialize arrays with values
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.5f;
        h_B[i] = i * 2.5f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Warm-up GPU
    {
        solve(d_A, d_B, d_C, N);
    }

    // 🧪 Time only GPU vector addition (no copy or alloc)
    {
        auto gpu_start = std::chrono::high_resolution_clock::now();
        solveN(d_A, d_B, d_C, N);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
        std::cout << "GPU Naive Kernel time: " << gpu_time.count() << " ms\n";
    }

    {
        auto gpu_start = std::chrono::high_resolution_clock::now();
        solve(d_A, d_B, d_C, N);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
        std::cout << "GPU Optimized Kernel time: " << gpu_time.count() << " ms\n";
    }

    // Copy result back from device to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 🧪 Time CPU vector addition
    {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        solveC(h_A, h_B, h_C_CPU, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
        std::cout << "CPU time: " << cpu_time.count() << " ms\n";
    }
    
    // Validate results - Compare C++ and CUDA output
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - h_C_CPU[i]) > 1e-5) {
            if (++errors <= 10)
                std::cout << "Mismatch at " << i << ": GPU " << h_C[i] << " CPU " << h_C_CPU[i] << "\n";
        }
    }
    
    if (errors == 0)
        std::cout << "Results match!\n";
    else
        std::cout << "Total mismatches: " << errors << "\n";

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_CPU;

    return 0;
}
