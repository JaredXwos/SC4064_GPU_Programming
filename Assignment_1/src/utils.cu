//
// Created by hatoa on 31/1/2026.
//
#include "utils.cuh"

#include <iostream>
#include <stdexcept>
#include <string>

// We probably? do not need real randomness? Because this is for benchmarking.
__global__ void fill(float* __restrict__ out, const int n, const float max_val)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t x = idx * 747796405u + 2891336453u; // LCG
    x ^= x >> 16; // Mix bits

    out[idx] = (1.0f / 16777216.0f) * max_val * (x & 0x00FFFFFF);
}

int parseSize(const int argc, const char* const argv[], const int index, const int defaultVal)
{
    if (argc <= index)
        return defaultVal;

    try {
        const int v = std::stoi(argv[index]);
        if (v <= 0)
            throw std::invalid_argument("non-positive");

        return v;
    }
    catch (...) {
        std::cerr << "Invalid integer argument at position "
                  << index << "\n";
        std::exit(1);
    }
}