#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// Define the block size for CUDA kernels (number of threads per block dimension)
#define BLOCK_SIZE 16

// Macro to check CUDA API call results and handle errors
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// --------------------------------------------------
// Naive GPU kernel: basic implementation of matrix multiplication
// Computes C = A * B
// Each thread computes one output element C[y][x]
// --------------------------------------------------
__global__ void matrix_mul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x; // Column index in C
    int y = blockDim.y * blockIdx.y + threadIdx.y; // Row index in C

    if ((x >= N) || (y >= M))
        return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[k + (y * K)] * B[x + (k * N)];
    }

    C[x + (y * N)] = sum;
}

// --------------------------------------------------
// Optimized GPU kernel: uses shared memory (tiling) for faster access
// --------------------------------------------------
__global__ void matrix_mul_shared(const float* A, const float* B, float* C, int M, int K, int N) {
    int tx = threadIdx.x; // Thread's local x within the block
    int ty = threadIdx.y; // Thread's local y within the block

    int x = blockDim.x * blockIdx.x + tx; // Global column index in C
    int y = blockDim.y * blockIdx.y + ty; // Global row index in C

    // Shared memory tiles for submatrices of A and B
    __shared__ float shA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shB[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // Loop over tiles of size BLOCK_SIZE in the K dimension
    for (int k = 0; k < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        // Load tile of A into shared memory
        if ((y < M) && (k * BLOCK_SIZE + tx) < K)
            shA[ty][tx] = A[(k * BLOCK_SIZE + tx) + (y * K)];
        else
            shA[ty][tx] = 0.0f;

        // Load tile of B into shared memory
        if ((x < N) && (k * BLOCK_SIZE + ty) < K)
            shB[ty][tx] = B[x + ((k * BLOCK_SIZE + ty) * N)];
        else
            shB[ty][tx] = 0.0f;

        __syncthreads(); // Synchronize to make sure tile is loaded

        // Compute partial product for this tile
        for (int tk = 0; tk < BLOCK_SIZE; ++tk) {
            sum += shA[ty][tk] * shB[tk][tx];
        }

        __syncthreads(); // Synchronize before loading new tile
    }

    // Store the final result in C
    if (x < N && y < M)
        C[x + (y * N)] = sum;
}

// --------------------------------------------------
// Utility function to measure GPU kernel execution time
// --------------------------------------------------
template <typename KernelFunc>
float measure_gpu_time(KernelFunc kernel) {
    cudaEvent_t start, stop;
    float ms = 0;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    kernel(); // Run the kernel lambda

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

// --------------------------------------------------
// Utility function to measure CPU execution time
// --------------------------------------------------
template <typename CpuFunc>
double measure_cpu_time(CpuFunc func) {
    auto start = std::chrono::high_resolution_clock::now();
    func(); // Run the CPU lambda
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// --------------------------------------------------
// MatrixMultiplier class encapsulates setup, execution, and validation
// --------------------------------------------------
class MatrixMultiplier {
public:
    MatrixMultiplier(int M, int K, int N)
        : M(M), K(K), N(N), sizeA(M * K), sizeB(K * N), sizeC(M * N) {
        // Allocate and initialize host matrices
        h_A = new float[sizeA];
        h_B = new float[sizeB];
        h_C = new float[sizeC];
        h_C_CPU = new float[sizeC];

        // Fill A and B with sample values
        for (int i = 0; i < sizeA; ++i)
            h_A[i] = static_cast<float>((i % 10) + 1);
        for (int i = 0; i < sizeB; ++i)
            h_B[i] = static_cast<float>(((i + 1) % 7) + 1);

        // Allocate device memory
        CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, sizeC * sizeof(float)));

        // Copy input matrices from host to device
        CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~MatrixMultiplier() {
        // Free host and device memory
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_C_CPU;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // Run both naive and optimized GPU kernels
    void runGpu() {
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Measure naive GPU kernel
        float t_naive = measure_gpu_time([&]() {
            matrix_mul_naive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
            CHECK_CUDA(cudaDeviceSynchronize());
        });
        std::cout << "GPU Naive Kernel time: " << t_naive << " ms\n";

        // Clear output matrix before optimized kernel
        CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(float)));

        // Measure optimized shared memory kernel
        float t_opt = measure_gpu_time([&]() {
            matrix_mul_shared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
            CHECK_CUDA(cudaDeviceSynchronize());
        });
        std::cout << "GPU Optimized Kernel time: " << t_opt << " ms\n";

        // Copy result from device to host
        CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Run CPU version of matrix multiplication
    void runCpu() {
        double t_cpu = measure_cpu_time([&]() {
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j) {
                    float sum = 0;
                    for (int k = 0; k < K; ++k)
                        sum += h_A[i * K + k] * h_B[k * N + j];
                    h_C_CPU[i * N + j] = sum;
                }
        });
        std::cout << "CPU time: " << t_cpu << " ms\n";
    }

    // Validate GPU result against CPU result
    void validate() const {
        int errors = 0;
        for (int i = 0; i < sizeC; ++i) {
            if (fabs(h_C[i] - h_C_CPU[i]) > 1e-4f) {
                if (++errors <= 10)
                    std::cout << "Mismatch at " << i << ": GPU " << h_C[i] << ", CPU " << h_C_CPU[i] << "\n";
            }
        }

        if (errors == 0)
            std::cout << "Results match!\n";
        else
            std::cout << "Total mismatches: " << errors << "\n";
    }

private:
    int M, K, N;              // Matrix dimensions
    int sizeA, sizeB, sizeC;  // Flattened sizes
    float *h_A, *h_B, *h_C, *h_C_CPU; // Host matrices
    float *d_A, *d_B, *d_C;           // Device matrices
};

// --------------------------------------------------
// Main function to run everything
// --------------------------------------------------
int main() {
    int M = 512, K = 512, N = 512; // Matrix dimensions: A(MxK), B(KxN), C(MxN)
    MatrixMultiplier mm(M, K, N);  // Create multiplier object

    mm.runGpu();   // Run and time GPU kernels
    mm.runCpu();   // Run and time CPU version
    mm.validate(); // Compare results

    return 0;
}
