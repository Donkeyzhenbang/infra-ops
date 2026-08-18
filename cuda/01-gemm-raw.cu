#include <stdio.h>
#include <cuda_runtime.h>


#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void sgemm_naive_cpu(float* A, float* B, float* C, int M, int N, int K)
{
    for(int x = 0; x < M; x ++){
        for(int y = 0; y < N ; y ++){
            float sum = 0.0f;
            for(int i = 0; i < K; i ++){
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N +y] = sum;
        }
    }
}


__global__ void sgemm_naive_kernel(float* A, float* B, float* C, int M, int N, int K)
{
    // A:M*K B:K*N C:M*N
    // threadIdx 所属block内编号
    // blockDim 每个block多少个线程
    // 一个线程处理一个C的元素
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < M && y < N){
        float sum = 0.0f;
        for(int i = 0; i < K; i++){
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }

}

void run_sgemm_naive(float *A, float *B, float *C, int m, int n, int k)
{
    dim3 block_size(32, 32);
    dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    sgemm_naive_kernel<<<grid_size, block_size>>>(A, B, C, m, n, k);
}

int main()
{
    int m = 256;
    int n = 256;
    int k = 256;

    // host memory
    float* h_A = (float*)malloc(m * k * sizeof(float));
    float* h_B = (float*)malloc(k * n * sizeof(float));
    float* h_C = (float*)malloc(m * n * sizeof(float));
    float* C_ref = new float[m * n];

    // initial matrix
    srand(42);
    for(int i = 0; i < m * k; i ++) h_A[i] = (float)(rand() % 100) / 10.0f;
    for(int i = 0; i < k * n; i ++) h_B[i] = (float)(rand() % 100) / 10.0f;
    
    // device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // copy data to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    run_sgemm_naive(d_A, d_B, d_C, m, n, k);
    // 等待 GPU 上所有之前提交的任务（kernel、拷贝等）全部执行完毕。
    cudaDeviceSynchronize(); 

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("kernel launch error : %s\n", cudaGetErrorString(err));
    }

    // copy data to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    sgemm_naive_cpu(h_A, h_B, C_ref, m, n, k);
    for(int i = 0; i < m * n; i ++){
        if(fabs(h_C[i] - C_ref[i]) > 1e-3){
            printf("Error: mismatch at index %d, expected %f, got %f\n", i, C_ref[i], h_C[i]);
            return 1;
        }
    }

    // destroy 
    free(h_A);
    free(h_B);
    free(h_C);
    h_A = nullptr;
    h_B = nullptr;
    h_C = nullptr;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    d_A = nullptr;
    d_B = nullptr;
    d_C = nullptr;

    return 0;

}