#ifndef GEMV_H
#define GEMV_H

// #include "math_functions.h"
#include "util.h"
#include "setting.h"
#include <loki/lokilib.h>
#include <loki/scratchpad.h>
#define MC2  32
#define KC2  64
#define NC2  128

#define MR2  4
#define NR2  1
#define MATRIX_NUM_CORE 8
//
//  Local buffers for storing panels from A, B and C
//
static Dtype _A_mv[MC2*KC2];
static Dtype _B_mv[KC2*NC2];
static Dtype _C_mv[MR2*NR2];


typedef struct gemv_data_ {
  int m;
  int n;
  int k;
  Dtype  *C;
  Dtype *B;
  Dtype *A;
  int     incRowC;
  int     incColC;
  int     incRowB;
  int     incColB;
  int     incRowA;
  int     incColA;
  int cores;
} gemv_data;

static void
pack_MRxk_mv(int k, const Dtype *A, int incRowA, int incColA,
          Dtype *buffer)
{
    Dtype* limit = &buffer[k*MR2];
    /*
    * the below is the original code
    */
    for (; buffer < limit; buffer += MR2) {
        // for (int i=0; i<MR; ++i) {
        //     buffer[i] = A[i*incRowA];
        // }
        
      asm volatile(
        "0: "
        "fetchr 1f \n"
        "ldw 0x0(%0) -> 10 \n"
        "addu r28, %0, %2 \n"
        "ldw 0x0(r28) -> 11 \n"
        "addu r28, r28, %2 \n"
        "ldw 0x0(r28) -> 12 \n"
        "addu r28, r28, %2 \n"
        "ldw 0x0(r28) -> 13 \n"

        "stw r4, 0x0(%1) -> 1 \n"
        "stw r5, 0x4(%1) -> 1 \n"
        "stw r6, 0x8(%1) -> 1 \n"
        "stw.eop r7, 0xc(%1) -> 1 \n"
        "1: \n"
        : "+&r"(A), "+&r"(buffer)
        : "r"(incRowA*4)
        : "r28"
        ); 

        A      += incColA;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A_mv(int mc, int kc, const Dtype *A, int incRowA, int incColA,
       Dtype *buffer)
{
    // fprintf(stderr, "pack A starts \n");
    int mp  = mc / MR2;
    int _mr = mc % MR2;

    int i, j;
    int core = get_core_id();
    buffer += kc*MR2*core;
    A += MR2*incRowA*core;
    for (i=core; i<mp; i+=MATRIX_NUM_CORE) {
        pack_MRxk_mv(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR2*MATRIX_NUM_CORE;
        A      += MR2*incRowA*MATRIX_NUM_CORE;
    }
    
    if ((_mr>0) && (core == mp % MATRIX_NUM_CORE)) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR2; ++i) {
                buffer[i] = 0;
            }
            buffer += MR2;
            A      += incColA;
        }
    }
    loki_tile_sync(MATRIX_NUM_CORE);
}



//
//  Micro kernel for multiplying panels from A and B.
//
static void
dgemm_micro_kernel_mv(int kc,
                   const Dtype *A, const Dtype *B,
                   Dtype *C)
{

    const Dtype* limit = &B[kc];
        asm volatile(
          "fetchr.eop 0f \n"
          "0: "
          "fetchr 1f \n"
          "or r15, r0, r0 \n"
          "or r16, r0, r0 \n"
          "or r17, r0, r0 \n"
          "or r18, r0, r0 \n"
          "addui %1, %1, -4 \n"
          "addui %2, %2, -4 \n"
          "addui.eop %3, %3, -4 \n"

          "1: "
          "addui %1, %1, 4 -> 10 \n"
          "addui %1, %1, 4 -> 11 \n"
          "addui %1, %1, 4 -> 12 \n"
          "addui %1, %1, 4 -> 13 \n"

          // "ldw 0x0(%2) -> 14 \n"
          "addui %2, %2, 4 -> 14 \n"

          "setlt.p r0, %2, %3 \n"     // compare if first limit has reached
          "or r24, r2, r0    \n"
          "psel.fetchr 1b, 2f\n"


          "mullw r30, r24, r4 \n"
          "addu r15, r15, r30 \n"
          "mullw r30, r24, r5 \n"
          "addu r16, r16, r30 \n"
          "mullw r30, r24, r6 \n"
          "addu r17, r17, r30 \n"
          "mullw r30, r24, r7 \n"
          "addu.eop r18, r18, r30 \n"

          "2: "
          "srai r15, r15, 0x0a \n"
          "srai r16, r16, 0x0a \n"
          "srai r17, r17, 0x0a \n"
          "srai r18, r18, 0x0a \n"

          "ldadd r15, %0, 0x00 -> 10 \n"
          "ldadd r16, %0, 0x04 -> 11 \n"
          "ldadd r17, %0, 0x08 -> 12 \n"
          "ldadd r18, %0, 0x0c -> 13 \n"
          "or r0, r4, r5 \n"
          "or r0, r6, r7 \n"
          "fetchr.eop 3f \n"
          "3:"
          : "+&r"(C)
          : "r"(A), "r"(B), "r"(limit)
          : "r15", "r16", "r17", "r18",
            "r20", "r21", "r22", "r23", 
            "r24", "r30"
          );

}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A_mv and _B_mv.
//
static void
dgemm_macro_kernel_mv(int     mc,
                   int     nc,
                   int     kc,
                   Dtype  *C,
                   Dtype  *B,
                   Dtype  *A)
{
    int mp = (mc+MR2-1) / MR2;

    int mr, nr;
    int i, j;
    int core = get_core_id();
    for (i=core; i<mp; i+=MATRIX_NUM_CORE) {
        dgemm_micro_kernel_mv(kc, &A[i*kc*MR2], 
                           B,
                           &C[i*MR2]);
    }
}



//
//  Compute C <- beta*C + A*B
//
void
dgemv_nn_worker(
         const void* data1)
{
    gemv_data* data = (gemv_data*)data1;
    int m = data->m;
    int n = data->n;
    int k = data->k;
    Dtype* A = data->A;
    Dtype* B = data->B;
    Dtype* C = data->C;
    int incRowA = data->incRowA;
    int incRowB = data->incRowB;
    int incRowC = data->incRowC;
    int incColA = data->incColA;
    int incColB = data->incColB;
    int incColC = data->incColC;

    int core = get_core_id();
    int addr = loki_mem_address(0, core, CH_REGISTER_4, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(10, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_5, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(11, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_6, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(12, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_7, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(13, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_2, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(14, addr);

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
            
              // core 0 transpose and pack the value
                pack_A_mv(mc, kc,
                       &A[i*MC2*incRowA+l*KC2*incColA], incRowA, incColA,
                       _A_mv);
              
              dgemm_macro_kernel_mv(mc, nc, kc,
                                 &C[i*MC2*incRowC+j*NC2],
                                 &B[l*KC2*incRowB+j*NC2*incColB],
                                 _A_mv
                                 );
        }
    }
    loki_tile_sync(MATRIX_NUM_CORE);
}




void
dgemv_nn(int            m,
         int            k,
         const Dtype   *A,
         int            incRowA,
         // int            incColA,
         const Dtype   *B,
         // int            incColB,
         // Dtype         beta,
         Dtype         *C
         // int            incColC
         ){

                  loki_init_default(MATRIX_NUM_CORE, 0);
                  gemv_data* data = malloc(sizeof(gemv_data));
                  data->m = m;
                  data->n = 1;
                  data->k = k;
                  data->C = C;
                  data->B = B;
                  data->A = A;
                  data->incRowA = incRowA;
                  data->incColA = 1;
                  data->incRowB = 1;
                  data->incColB = 1;
                  data->incRowC = 1;
                  data->incColC = 1;
                  data->cores = MATRIX_NUM_CORE;
                  
                  distributed_func* config = malloc(sizeof(distributed_func));
                  config->cores = MATRIX_NUM_CORE;
                  // set the function to be macro_kernel
                  config->func = &dgemv_nn_worker;
                  config->data = data;
                  config->data_size = sizeof(gemv_data);
                  loki_execute(config);
}

#endif
