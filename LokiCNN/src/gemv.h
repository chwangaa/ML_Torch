/*! \file gemv.h
    \brief none-assembly version of GEMV algorithm for FIXED-POINT datatype
*/
#ifndef GEMV_H
#define GEMV_H
#include "math_functions.h"
/// packed height of A
#define MC2  256
/// packed width of A
#define KC2  512
#define NC2  1024

#define MR2  4
#define NR2  1

//
//  Local buffers for storing panels from A, B and C
//
/// storage for packed A
static Dtype _A_mv[MC2*KC2];

//
//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk_mv(int k, const Dtype *A, int incRowA,
          Dtype *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR2; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR2;
        A      += 1;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A_mv(int mc, int kc, const Dtype *A, int incRowA,
       Dtype *buffer)
{
    int mp  = mc / MR2;
    int _mr = mc % MR2;

    int i, j;

    for (i=0; i<mp; ++i) {
        pack_MRxk_mv(kc, A, incRowA, buffer);
        buffer += kc*MR2;
        A      += MR2*incRowA;
    }
    if (_mr>0) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR2; ++i) {
                buffer[i] = 0.0;
            }
            buffer += MR2;
            A      += 1;
        }
    }
}



//
//  Micro kernel for multiplying panels from A and B.
//
static void
dgemm_micro_kernel_mv(int kc,
                   const Dtype *A, const Dtype *B,
                   Dtype *C)
{

    int i, j, l;

    const Dtype* limit = &B[kc];
    for (; B < limit; B++) {
        for (i=0; i<MR2; ++i) {
            C[i] = add_multiply(C[i], A[i], *B);
        }
        A += MR2;
    }

}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A_mv and _B_mv.
//
static void
dgemm_macro_kernel_mv(int     mc,
                   int     kc,
                   Dtype  *C,
                   Dtype  *B)
{
    int mp = (mc+MR2-1) / MR2;


    int mr, nr;
    int i, j;

    for (i=0; i<mp; ++i) {
        dgemm_micro_kernel_mv(kc, &_A_mv[i*kc*MR2], 
                           &B[j*kc*NR2],
                           &C[i*MR2+j]);
    }
}


///  Compute C += A*b
void
dgemv_nn(int            m,
         int            k,
         const Dtype   *A,
         int            incRowA,
         const Dtype   *B,
         Dtype         *C
         )
{

    int mb = (m+MC2-1) / MC2;
    int kb = (k+KC2-1) / KC2;

    int _mc = m % MC2;
    int _kc = k % KC2;

    int mc, nc, kc;
    int i, j, l;

    for (l=0; l<kb; ++l) {
        kc    = (l!=kb-1 || _kc==0) ? KC2   : _kc;

        for (i=0; i<mb; ++i) {
            mc = (i!=mb-1 || _mc==0) ? MC2 : _mc;

            pack_A_mv(mc, kc,
                   &A[i*MC2*incRowA+l*KC2], incRowA,
                   _A_mv);

            dgemm_macro_kernel_mv(mc, kc,
                               &C[i*MC2+j*NC2],
                               &B[l*KC2+j*NC2]);
        }
    }
}

#endif