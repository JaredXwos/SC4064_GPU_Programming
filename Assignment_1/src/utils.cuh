//
// Created by hatoa on 31/1/2026.
//

#ifndef SC4046TESTBED_RANDOM_FILL_CUH
#define SC4046TESTBED_RANDOM_FILL_CUH
#include <cuda/pipeline>
__host__ __device__ __forceinline__
constexpr unsigned flatten(const unsigned row, const unsigned column, const unsigned width) {
    return row * width + column;
}

// Simple kernel to fill arr with LCG based random numbers
__global__ void fill(float* __restrict__ out, int n, float max_val);

int parseSize(int argc, const char* const argv[], int index, int defaultVal);
#endif //SC4046TESTBED_RANDOM_FILL_CUH