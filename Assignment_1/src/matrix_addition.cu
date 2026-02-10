#include <iomanip>
#include <iostream>

#include "utils.cuh"

__global__ void matrix_add_1d(
    const unsigned row_count,
    const unsigned column_count,
    const float *__restrict__ in1,
    const float *__restrict__ in2,
    float *__restrict__ out) {
    if (const auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < row_count * column_count)
        // Thread (i * j) % 1024 in block (i * j + 1023) / 1024 handles element (i,j)
        out[idx] = in1[idx] + in2[idx];
}
__global__ void matrix_add_2d(
    const unsigned row_count,
    const unsigned column_count,
    const float *__restrict__ in1,
    const float *__restrict__ in2,
    float *__restrict__ out){

    const auto r = blockIdx.y * blockDim.y + threadIdx.y;
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < row_count && c < column_count)
    {
        // Thread (i % 32, j % 32), in block ( (i+31)/32, (j+31)/32 ) handles element (i,j)
        const auto idx = r * row_count + c;
        out[idx] = in1[idx] + in2[idx];
    }
}

#define ROW 8192
#define COLUMN 8192

int main() {

    constexpr size_t arrLength = ROW * COLUMN;
    constexpr size_t arrSize = arrLength * sizeof(float);
    constexpr size_t blockSize1D = 1024; //Simple addition has low register pressure.
    constexpr size_t blockCount1D = (arrLength + blockSize1D - 1) / blockSize1D;
    constexpr auto blockSize2D = dim3(32, 32); //Align with 1024
    constexpr auto blockCount2D = dim3(ROW/32, COLUMN/32);

    cudaEvent_t start, stop;
    float ms_1d = 0, ms_2d = 0;
    // I suppose in benchmarking we shouldn't use thrust vectors?
    float *A, *B, *C;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&A, arrSize);
    cudaMalloc(&B, arrSize);
    cudaMalloc(&C, arrSize);

    fill<<<blockCount1D, blockSize1D>>>(A, arrLength, 100);
    fill<<<blockCount1D, blockSize1D>>>(B, arrLength, 100);

    //Test the 1D Grid/Block
    cudaMemset(C, 0, arrSize);

    cudaEventRecord(start);
    matrix_add_1d<<<blockCount1D, blockSize1D>>>(ROW, COLUMN, A, B, C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_1d, start, stop);

    // Test the 2D Grid/Block
    cudaMemset(C, 0, arrSize);

    cudaEventRecord(start);
    matrix_add_2d<<<blockCount2D, blockSize2D>>>(ROW, COLUMN, A, B, C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_2d, start, stop);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const auto flops_1d = static_cast<const double>(arrLength) / (ms_1d / 1000.0);
    const auto flops_2d = static_cast<const double>(arrLength) / (ms_2d / 1000.0);

    std::cout
        << std::fixed
        << "1D: "
        << std::setprecision(5) << ms_1d << " ms"
        << " | " << std::setprecision(5) << (flops_1d / 1e9) << " GFLOPS"
        << std::endl

        << "2D: "
        << std::setprecision(5) << ms_2d << " ms"
        << " | " << std::setprecision(5) << (flops_2d / 1e9) << " GFLOPS"
        << std::endl

        << std::setprecision(3)
        << "Speedup (2D vs 1D): " << ms_1d / ms_2d << "x | "
        << "Percent improvement: " << (ms_1d - ms_2d) / ms_1d * 100.0 << " | "
        << "Throughput ratio: " << flops_2d / flops_1d << "x"
        << std::endl
        << std::defaultfloat;

    return 0;
}
