#include <iomanip>
#include <iostream>
#include <string>

#include "utils.cuh"

#define CUDA_CHECK(call) do {\
    cudaError_t err = (call); \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR: %s (at %s: %d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(err); \
    } \
} while(0)

// Not specified, but assumed outplace because subsequent questions specify outplace.
__global__ void add_outplace(
    const int length,
    const float *__restrict__ arr1,
    const float *__restrict__ arr2,
    float *__restrict__ out) {
    if (const auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length)
        out[idx] = arr1[idx] + arr2[idx];
}

#define LENGTH (1<<30) // still smaller than INT_MAX, so int indexing is acceptable

int main(const int argc, const char* const argv[]) {
    cudaEvent_t start, stop;
    float ms = 0; //event time in miliseconds
    float *arr1, *arr2, *result; //because cuda_check is required we cannot use thrust arrays

    constexpr size_t arrSize = LENGTH * sizeof(float);
    const unsigned blockSize = parseSize(argc, argv, 1, 0);
    if (blockSize == 0) {
        std::cerr << "Please specify block size." << std::endl;
        return 1;
    }
    const unsigned blockCount = (LENGTH + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMalloc(&arr1, arrSize));
    CUDA_CHECK(cudaMalloc(&arr2, arrSize));
    CUDA_CHECK(cudaMalloc(&result, arrSize));

    // Both these two fills can be parallel so 2x device sync isn't strictly necessary
    // However the task is to use the error macro
    // So to capture the errors of both kernel runs separately we do it serially and sync device each time
    fill<<<blockCount, blockSize>>>(arr1, LENGTH, 100.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fill<<<blockCount, blockSize>>>(arr2, LENGTH, 100.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    add_outplace<<<blockCount, blockSize>>>(LENGTH, arr1, arr2, result);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaFree(arr1));
    CUDA_CHECK(cudaFree(arr2));
    CUDA_CHECK(cudaFree(result));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const auto flops = static_cast<const double>(LENGTH) / (ms / 1000.0);
    std::cout
        << std::fixed
        << "Block " << std::setw(3) << blockSize
        << " | " << std::setprecision(3) << ms << " ms"
        << " | " << std::setprecision(2) << (flops / 1e9) << " GFLOPS"
        << std::endl
        << std::defaultfloat;
    return 0;
}