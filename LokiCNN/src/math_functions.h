/*! \file math_functions.h
    \brief various math functions including gemm, gemv, etc
*/
// Convolutional Layer --------------------------------------------------------
#ifndef MATH_LIB_H
#define MATH_LIB_H

#include "setting.h"

#ifdef LOKI
  #include "im2col_loki.h"
#else
  #include <cblas.h>
  #include "im2col.h"
#endif

/* define arithmetic for the defined data-type */
#if defined(FIX16)
  #include <fix16_loki.h>
#elif defined(FIX8)
  #include <fix8_loki.h>
#else
inline storage_t multiply(storage_t d1, storage_t d2) {return d1*d2;}
inline storage_t add(storage_t d1, storage_t d2) {return d1+d2;}
inline storage_t divide(storage_t d1, storage_t d2) {return d1/d2;}
inline storage_t exp_t(storage_t input) { return exp(input); }
inline storage_t readDouble(double input) { return input; }
inline storage_t add_multiply(storage_t s1, storage_t m1, storage_t m2){ return s1 + m1*m2;}
#endif

/*! \fn void assign_bias(const int layer, const int layer_size, storage_t* biases, storage_t* outputs)
    \brief initialize output with the given biases
    \param layer the number of biases, and the number of layers in outputs
    \param layer_size the size of output should be layer*layer_size
    \param biases an array of size layer, the first layer_size pixels of output will be assigned the first term of biases, and so on
    \param outputs an flat array corresponding to the output image
*/
void assign_bias(const int layer, const int layer_size, storage_t* biases, storage_t* outputs){
  for(int d = 0; d < layer; d++){
    storage_t bias = biases[d];
    for(int j = 0; j < layer_size; j++){
      outputs[j+layer_size*d] = bias;
    }
  }
}

/*! 
    \brief matrix multiplication routine, performs C += A*B
    \param M height of A and C
    \param N width of B and C
    \param K width of A, height of B
    \param A input matrix A
    \param incRowA specifies how to find the next row, this add flexbility to get to the next row, i.e. after taking K entries, there can be a gap between the next row
    \param B input matrix B
    \param incRowB specifies how to find the next row in B
    \param C output matrix C
    \param incRowC specifies how to find the next row in C

    note that for our purpose, incRowA, incRowB, incRowC may be redundant
    in cblas, there are also incColA, incColB, incColC, performance in Loki increases a little when eliminating some unnecessary variables
*/
inline void cblas_gemm(int M, int N, int K,
                       const Dtype *A, int incRowA,
                       const Dtype *B, int incRowB,
                       Dtype *C, int incRowC);

/*! 
    \brief matrix-vector multiplication routine, performs C += A*b
    \param M height of A and C
    \param K width of A, height of B
    \param A input matrix A
    \param incRowA specifies how to find the next row, this add flexbility to get to the next row, i.e. after taking K entries, there can be a gap between the next row
    \param B input vector B
    \param incRowB specifies how to find the next row in B
    \param C output matrix C
    \param incRowC specifies how to find the next row in C
*/
inline void cblas_gemv(int M, int K,
                        const Dtype *A, int incRowA,
                        const Dtype *B,
                        Dtype *C);


// define cblas_gemm routine 
#if ! defined GPU
  #if defined DOUBLE && !defined LOKI
    	inline void cblas_gemm(int M, int N, int K,
           			const Dtype *A,
           			int incRowA,
           			const Dtype *B,
           			int incRowB,
           			Dtype *C,
           			int incRowC){
    		return cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    			M, N, K,
    			1, 
    			A, K, 
    			B, N, 
    			1, 
    			C, N);
    	}

      inline void cblas_gemv(int M, int K,
              const Dtype *A,
              int incRowA,
              const Dtype *B,
              Dtype *C){
        return cblas_dgemv(CblasRowMajor, CblasNoTrans,
          M, K,
          1, 
          A, K, 
          B, 1, 
          1, 
          C, 1);
      }



  #elif defined FLOAT && !defined LOKI
    	inline void cblas_gemm(int M, int N, int K,
           			const Dtype *A,
           			int incRowA,
           			const Dtype *B,
           			int incRowB,
           			Dtype *C,
           			int incRowC){
    		return cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    			M, N, K,
    			1, 
    			A, K, 
    			B, N, 
    			1, 
    			C, N);
    	}

      inline void cblas_gemv(int M, int K,
              const Dtype *A,
              int incRowA,
              const Dtype *B,
              Dtype *C){
        return cblas_sgemv(CblasRowMajor, CblasNoTrans,
          M, K,
          1, 
          A, K, 
          B, 1, 
          1, 
          C, 1);
      }

  #else

    	#if defined LOKI && defined FIX8
    		#include "gemm_loki.h"
        #include "gemv_loki.h"
    	#else
    		#include "gemm.h"
        #include "gemv.h"
    	#endif
    	  	
    	  	inline void cblas_gemm(int M, int N, int K,
           			const Dtype *A, int incRowA,
           			const Dtype *B, int incRowB,
           			Dtype *C, int incRowC){
          	return dgemm_nn(M, N, K, 
                  A, K, 
                  B, N, 
                  C, N);
    		    }

          inline void cblas_gemv(int M, int K,
                  const Dtype *A,
                  int incRowA,
                  const Dtype *B,
                  Dtype *C){
            dgemv_nn(
              M, K,
              A, incRowA, 
              B,
              C);
          }

  #endif

#endif

#if defined GPU

    #include "gemv.h"
    #include "gemm_gpu.h"
    inline void cblas_gemm(int M, int N, int K,
          const Dtype *A, int incRowA,
          const Dtype *B, int incRowB,
          Dtype *C, int incRowC){
      return gemm_gpu(M, N, K,
                      A, K,
                      B, N,
                      C, N);
    }

    inline void cblas_gemv(int M, int K,
            const Dtype *A,
            int incRowA,
            const Dtype *B,
            Dtype *C){
      dgemv_nn(
        M, K,
        A, incRowA, 
        B,
        C);
    }
#endif


#endif