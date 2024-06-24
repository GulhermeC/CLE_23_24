/**
 *   Guilherme Craveiro, May 2024
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include <cuda_runtime.h>

/* allusion to internal functions */

static double get_delta_time(void); //medir tempos

int main(int argc, char **argv)
{

    /* set up the device */

    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties (&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice (dev)); 
}

static double get_delta_time(void){
    static struct timespec t0,t1;

    t0 = t1;
    if(clock_gettime(CLOCK_MONOTONIC,&t1) != 0)
    {
        perror("clock_gettime");
        exit(1);
    }
    return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}