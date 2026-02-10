#include <iomanip>
#include <iostream>
#include <ostream>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <bit>

#include "utils.cuh"
// If we do not template blockDim.x and blockDim.y it cannot pragma unroll at runtime.
template <unsigned block_x, unsigned block_y>
__global__ void matrix_multiply(
    const unsigned m, const unsigned n, const unsigned k,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out)
{
    extern __shared__ float smem[];

    constexpr unsigned k4_step = block_x < block_y ? block_x : block_y; // The k as per float4
    constexpr unsigned k_step = k4_step << 2; // The k, as per float

    // We will be double buffering, so we can load while compute happens
    // A tiles are blockDim.y rows of k_step.
    // B tiles are k_step rows of blockDim.x
    auto (*As)[block_y][k_step] = reinterpret_cast<float (*)[block_y][k_step]>(smem);
    auto (*Bs)[k_step][block_x] = reinterpret_cast<float (*)[k_step][block_x]>(As + 2);

    // To increase tile size, we load a float4 instead
    auto (*As4)[block_y][k4_step] = reinterpret_cast<float4 (*)[block_y][k4_step]>(As);
    auto (*Bs4)[k_step][block_x >> 2] = reinterpret_cast<float4 (*)[k_step][block_x >> 2]>(Bs);
    const auto *A4 = reinterpret_cast<const float4*>(A);
    const auto *B4 = reinterpret_cast<const float4*>(B);


    // Each tile has an m_value from 0 to M and an n_value from 0 to N.
    const unsigned m_value = blockIdx.y * block_y + threadIdx.y;
    const unsigned n_value = blockIdx.x * block_x + threadIdx.x;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f, acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;

    unsigned buf = 0;
    // Preload the first tile
    if (m_value < m && threadIdx.x < k4_step)
        As4[buf][threadIdx.y][threadIdx.x] = A4[m_value * (k >> 2) + threadIdx.x];

    const unsigned b_row  = (threadIdx.y << 2) + threadIdx.x / (block_x >> 2);
    const unsigned b_col4 = threadIdx.x % (block_x >> 2);
    if (threadIdx.y < k4_step && blockIdx.x * (block_x >> 2) + b_col4 < n)
        Bs4[buf][b_row][b_col4] = B4[b_row * (n >> 2) + blockIdx.x * (block_x >> 2) + b_col4];


    for (unsigned tile_begin = k_step; tile_begin <= k; tile_begin += k_step)
    {
        __syncthreads();
        const unsigned next = buf ^ 1;

        // preload next tile (if exists)
        if (tile_begin < k)
        {
            if (m_value < m && threadIdx.x < k4_step)
                As4[next][threadIdx.y][threadIdx.x] = A4[m_value * (k >> 2) + threadIdx.x + (tile_begin >> 2)];
            if (threadIdx.y < k4_step && blockIdx.x * (block_x >> 2) + b_col4 < n)
                Bs4[next][b_row][b_col4] = B4[(b_row + (tile_begin >> 2)) * (n >> 2) + blockIdx.x * (block_x >> 2) + b_col4];
        }

        #pragma unroll
        for (unsigned kk = 0; kk < k_step; kk += 8) {
            acc0 += As[buf][threadIdx.y][kk + 0] * Bs[buf][kk + 0][threadIdx.x];
            acc1 += As[buf][threadIdx.y][kk + 1] * Bs[buf][kk + 1][threadIdx.x];
            acc2 += As[buf][threadIdx.y][kk + 2] * Bs[buf][kk + 2][threadIdx.x];
            acc3 += As[buf][threadIdx.y][kk + 3] * Bs[buf][kk + 3][threadIdx.x];
            acc4 += As[buf][threadIdx.y][kk + 4] * Bs[buf][kk + 4][threadIdx.x];
            acc5 += As[buf][threadIdx.y][kk + 5] * Bs[buf][kk + 5][threadIdx.x];
            acc6 += As[buf][threadIdx.y][kk + 6] * Bs[buf][kk + 6][threadIdx.x];
            acc7 += As[buf][threadIdx.y][kk + 7] * Bs[buf][kk + 7][threadIdx.x];
        }
        buf = next;
    }
    if (m_value < m && n_value < n) out[m_value * n + n_value] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
}

// We cannot function pointer a cuda kernel so we need to wrap it in something we can
template<int tile_x, int tile_y>
void launch(dim3 grid, size_t smem,
            const unsigned m, const unsigned n, const unsigned k,
            const float* A, const float* B, float* C)
{
    if (tile_y * tile_x == 1024) cudaFuncSetAttribute(
            matrix_multiply<tile_x, tile_y>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            65536
        ); // Apparently the default shared memory size is too small
    dim3 block(tile_x, tile_y);
    matrix_multiply<tile_x, tile_y><<<grid, block, smem>>>(m,n,k,A,B,C);
}
// Need to generate every possible kernel as a table for dynamic calling of the right constexpr template
using Fn = void(*)(dim3, size_t, unsigned, unsigned, unsigned, const float*, const float*, float*);
static constexpr Fn table[9][9] = {
    {nullptr, nullptr, nullptr,
        nullptr,nullptr, nullptr,
        nullptr, nullptr, nullptr
    }, // 1
    {nullptr, nullptr, nullptr,
        nullptr,nullptr, nullptr,
        nullptr, nullptr, nullptr
    }, // 2
    {
        nullptr, nullptr, launch<4,4>,
        launch<4,8>,  launch<4,16>,  launch<4,32>,
        launch<4,64>, launch<4,128>, launch<4,256>
    },
    {
        nullptr, nullptr, launch<8,4>,
        launch<8,8>,  launch<8,16>,  launch<8,32>,
        launch<8,64>, launch<8,128>, nullptr
    },
    {
        nullptr, nullptr, launch<16,4>,
        launch<16,8>, launch<16,16>, launch<16,32>,
        launch<16,64>, nullptr, nullptr
    },
    {
        nullptr, nullptr, launch<32,4>,
        launch<32,8>, launch<32,16>, launch<32,32>,
        nullptr, nullptr, nullptr},
    {
        nullptr, nullptr, launch<64,4>,
        launch<64,8>, launch<64,16>, nullptr,
        nullptr, nullptr, nullptr
    },
    {
        nullptr, nullptr, launch<128,4>,
        launch<128,8>, nullptr, nullptr,
        nullptr, nullptr, nullptr
    },
    {
        nullptr, nullptr, launch<256,4>,
        nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr
    },
};

#define M 8192
#define K 8192
#define N 8192
int main(const int argc, const char* const argv[])  {

    constexpr int A_length = M * K, B_length = K * N, C_length = M * N;
    constexpr int A_blockCount = (A_length+1023)/1024, B_blockCount = B_length/1024;

    const auto block = dim3(
        parseSize(argc, argv, 1, 0),
        parseSize(argc, argv, 2, 0));
    if (block.x == 0 || block.y == 0) {
        std::cerr << "Please specify block x and y sizes" << std::endl;
        return 1;
    }
    if (block.x * block.y > 1024) {
        std::cerr << "Block size is too big" << std::endl;
        return 1;
    }

    const auto fn = table[std::countr_zero(block.x)][std::countr_zero(block.y)];
    if (!fn) throw std::runtime_error("unsupported tile");

    const auto grid = dim3(
    (N + block.x - 1) / block.x,
    (M + block.y - 1) / block.y);

    const auto memory = 4 * block.x * block.y * sizeof(float4);

    cudaEvent_t start, stop;
    float ms = 0;
    float *A, *B, *C, *C_ref;;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&A, A_length * sizeof(float));
    cudaMalloc(&B, B_length * sizeof(float));
    cudaMalloc(&C, C_length * sizeof(float));
    cudaMalloc(&C_ref, C_length * sizeof(float));

    fill<<<A_blockCount, 1024>>>(A, A_length, 1);
    fill<<<B_blockCount, 1024>>>(B, B_length, 1);
    cudaMemset(C, 0, C_length * sizeof(float));

    cudaEventRecord(start);
    fn(grid, memory, M,N,K,A,B,C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    std::cout
        << std::fixed
        << "Block " << std::setw(3) << block.x << "x" << std::setw(3) << block.y
        << " (" << block.x * block.y << " threads)"
        << " | " << std::setprecision(3) << ms << " ms"
        << " | " << std::setprecision(2) << 2.0 * M * N * K / (ms / 1000.0) / 1e9 << " GFLOPS"
        << std::endl
        << std::defaultfloat;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
