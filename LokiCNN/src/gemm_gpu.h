#include <math.h>
#include "cuda.h"
#include "setting.h"
void gemm_ongpu(int M, int N, int K, 
        Dtype *A_gpu, int lda, 
        Dtype *B_gpu, int ldb,
        Dtype *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    /* need cublasDgemm for double, cublasSgemm for float */
    cudaError_t status = cublasSgemm(handle, 
                                     CUBLAS_OP_N, CUBLAS_OP_N, 
                                     N, M, K, 
                                     1,
                                     B_gpu, ldb, 
                                     A_gpu, lda, 
                                     1, 
                                     C_gpu, ldc);
    check_error(status);
}

void gemm_gpu(int M, int N, int K, 
        Dtype *A, int lda, 
        Dtype *B, int ldb,
        Dtype *C, int ldc)
{
    Dtype *A_gpu = cuda_make_array(A, lda*M);
    Dtype *B_gpu = cuda_make_array(B, ldb*K);
    Dtype *C_gpu = cuda_make_array(C, ldc*M);

    gemm_ongpu(M, N, K, A_gpu, lda, B_gpu, ldb, C_gpu, ldc);

    cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free(A_gpu);
    cuda_free(B_gpu);
    cuda_free(C_gpu);
}