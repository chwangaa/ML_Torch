#ifndef CUDA_H
#define CUDA_H
int gpu_index = 0;
#define BLOCK 256

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "assert.h"
#include "setting.h"
#include <stdlib.h>
#include <time.h>


void check_error(cudaError_t status)
{
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
}


cublasHandle_t blas_handle()
{
    static int init = 0;
    static cublasHandle_t handle;
    if(!init) {
        cublasCreate(&handle);
        init = 1;
    }
    return handle;
}

Dtype *cuda_make_array(Dtype *x, int n)
{
    Dtype *x_gpu;
    unsigned long size = sizeof(Dtype)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    return x_gpu;
}


void cuda_free(Dtype *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(Dtype *x_gpu, Dtype *x, int n)
{
    unsigned long size = sizeof(Dtype)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void cuda_pull_array(Dtype *x_gpu, Dtype *x, int n)
{
    unsigned long size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

#endif