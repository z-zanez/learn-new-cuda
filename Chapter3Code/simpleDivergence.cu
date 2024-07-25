#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void mathKernel1(float* C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    /*if(tid % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }*/

    C[tid] = a + b;
}

__global__ void mathKernel2(float* C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if(tid / 32 % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }

    C[tid] = a + b;
}

__global__ void mathKernel3(float* C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    bool ipred = (tid % 2 == 0);
    if(ipred)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }

    C[tid] = a + b;
}

__global__ void warmingup(float* C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if(tid / 32 % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }

    C[tid] = a + b;
}

int main(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    // set up data size
    int size = 64;
    int blocksize = 64;
    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size = atoi(argv[2]);

    // set up execution configuration
    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1) / block.x, 1);
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

    // allocate gpu memory
    float* d_C;
    int nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    // run a warmup kernel to remove overhead
    double iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    warmingup<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("warmup <<<%4d, %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    // run kernel 1
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<<%4d, %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    // run kernel 2
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel2 <<<%4d, %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    // run kernel 3
    iStart = cpuSecond();
    mathKernel3<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel3 <<<%4d, %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    // free gpu memory and reset device
    cudaFree(d_C);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}