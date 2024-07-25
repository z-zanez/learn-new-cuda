#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// 并行归约
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n)
    {
        return;
    }

    // in-place reduction in gloabl memory
    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if(tid % (2 * stride) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}
// 改善并行归约的分化
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n)
    {
        return;
    }

    // in-place reduction in gloabl memory
    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        if(index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// 交错配对归约
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n)
    {
        return;
    }

    // in-place reduction in gloabl memory
   for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to gloabl memory
    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// 交错配对的展开归约
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // boundary check
    if(idx + blockDim.x < n)
    {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();

    // in-place reduction in gloabl memory
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to gloabl memory
    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if(idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    // in-place reduction in gloabl memory
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to gloabl memory
    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }

}

// 展开线程的归约
__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if(idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    // in-place reduction in gloabl memory
    for(int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // unrolling warp
    if(tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // write result for this block to gloabl memory
    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }

}

__global__ void warmup(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n)
    {
        return;
    }

    // in-place reduction in gloabl memory
    for(int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if(tid % (2 * stride) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1)
        return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

int main(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d\n", size);

    // execution configuration
    int blocksize = 512; // initial block size
    if(argc > 1)
    {
        blocksize = atoi(argv[1]); // block size from command line argument
    }
    dim3 block(blocksize, 1); // block size
    dim3 grid((size + block.x - 1) / block.x, 1); // grid size
    printf("    grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    // initialize the array
    for(int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((int**)&d_idata, bytes);
    cudaMalloc((int**)&d_odata, grid.x * sizeof(int));

    // cpu reduction
    iStart = cpuSecond();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: warmup
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    //printf("gpu warmup elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduceNeighbored elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceNeighboredLess
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduceNeighboredLess elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceInterleaved
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduceInterleaved elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 5: reduceUnrolling2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x / 2; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduceUnrolling2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 6: reduceUnrollWarps8
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x / 8; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduceUnrollWarps8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);

    // reset device
    cudaDeviceReset();

    // check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult)
    {
        printf("Test failed!\n");
    }
    return EXIT_SUCCESS;
}