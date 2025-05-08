# CUDA Projects

This repository contains a collection of algorithms implemented using **C++** and **CUDA**. Each project consists of both C++ and CUDA implementations of various algorithms. The repository also includes performance comparisons, where the **C++ implementation** is compared with the **CUDA kernel** in terms of both correctness and execution time.

## Features

* **C++ and CUDA Implementations**: For each algorithm, both a CPU (C++) and GPU (CUDA) implementation are provided to allow performance comparison between the two approaches.
* **Performance Comparison**: The time taken by the C++ and CUDA implementations is measured and displayed to highlight the performance improvements (or bottlenecks) when leveraging GPU acceleration.
* **Algorithm Variety**: A wide range of algorithms from different domains, such as **linear algebra**, **image processing**, and **searching** algorithms, have been implemented.
* **Result Verification**: After executing the algorithms, the results of the C++ and CUDA versions are compared to ensure correctness.

## Projects

Each project in this repository follows this general structure:

1. **Problem Description**: The problem statement and its constraints.
2. **C++ Implementation**: The algorithm is implemented using standard C++.
3. **CUDA Implementation**: The algorithm is implemented using CUDA for parallel execution on the GPU.
4. **Timing and Comparison**: Performance timings for both implementations, along with an accuracy check to verify if the results from C++ and CUDA are identical.
5. **Optimization**: Explanation of optimizations applied in the CUDA kernel to improve performance.

Online Platform to execute: https://leetgpu.com/playground

Note : Portions of the basic coding framework, including main(), were derived with support from ChatGPT. 
