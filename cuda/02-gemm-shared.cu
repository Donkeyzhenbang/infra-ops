#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void sgemm_naive_cpu(float *A, float *B, float *C, int M, int N, int K)
{
    for (int x = 0; x < M; x++)
    {
        for (int y = 0; y < N; y++)
        {
            float sum = 0.0f;
            for (int i = 0; i < K; i++)
            {
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = sum;
        }
    }
}

/**
 * 一个 block 负责 BLOCKSIZE * BLOCKSIZE 的 C tile
 * 一个线程负责 C tile 一个元素
 * 外层循环装入一块 A B 到 shared memory
 * 同步后 每个线程取 A_shared一行 B_shared一列 做点积
 * 跨多个tile把结果累加到寄存器标量tmp中
 * 循环结束 每个线程把 tmp 写回自己 C 元素
 */
template<const int BLOCKSIZE>
__global__ void sgemm_shared_mem_kernel(float *A, float *B, float *C, int m, int n, int k)
{
    // 启动时候确定了 grid block 形状
    // c_row 就是m / BLOCKSIZE 做除法结果
    const uint c_row = blockIdx.x;
    const uint c_col = blockIdx.y;

    __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

    // grid 是一维
    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_col = threadIdx.x % BLOCKSIZE;

    // 移动指针到相对位置
    A += c_row * BLOCKSIZE * K;
    B += c_col * BLOCKSIZE;
    C += c_row * BLOCKSIZE * N + c_col * BLOCKSIZE;

    float tmp = 0.0f;
    for(int i = 0; i < K; i += BLOCKSIZE){ 
        // 外循环处理一个tile 内循环处理tile每一个元素
        A_shared[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
        B_shared[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];
        // wait for all threads to finish loading
        __syncthreads();

        for(int j = 0; j < BLOCKSIZE; j ++){
            tmp += A_shared[thread_row * BLOCKSIZE + j] * B_shared[j * BLOCKSIZE + thread_col];
        }
        // wait for all threads to finish computing
        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

    }
    C[thread_row * N + thread_col] = tmp;

}


void run_sgemm_shared_memory(float *A, float *B, float *C, int m, int n, int k)
{
    const int BLOCKSIZE = 32;
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);
    dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
    sgemm_shared_mem_kernel<BLOCKSIZE><<<block_size, grid_size>>>(A, B, C, m, n, k);
}
