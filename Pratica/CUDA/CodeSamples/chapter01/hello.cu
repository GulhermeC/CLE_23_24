#include "../common/common.h"
#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU()
{
    printf("Hello World from GPU (%d, %d, %d) - (%d, %d, %d)!\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char **argv)
{
    dim3 grid, block;

    // {x, y, z} - Dimensions
    grid = {3, 2, 1};
    block = {3, 3, 1};

    printf("Hello World from CPU!\n");

    // helloFromGPU<<<1, 10>>>();
    helloFromGPU<<<grid, block>>>();
    CHECK(cudaDeviceReset());
    return 0;
}


