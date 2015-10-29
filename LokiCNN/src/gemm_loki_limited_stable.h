#ifndef GEMM_H
#define GEMM_H
#include <loki/lokilib.h>
#include <loki/scratchpad.h>
#include "setting.h"
#include "util.h"

#define MC  32
#define KC  64
#define NC  128

#define MR  4
#define NR  4
//
//  Local buffers for storing panels from A, B and C
//
static Dtype _A[MC*KC] __attribute__((aligned(0x1000)));
static Dtype _B[KC*NC] __attribute__((aligned(0x1000)));
//
//  Packing complete panels from A (i.e. without padding)
//


static void
pack_MRxk(int k, const Dtype *A, int incRowA, int incColA,
          Dtype *buffer)
{
    Dtype* limit = &buffer[k*MR];
    /*
    * the below is the original code
    */
    for (; buffer < limit; buffer += MR) {
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
pack_A(int mc, int kc, const Dtype *A, int incRowA, int incColA,
       Dtype *buffer, int MATRIX_NUM_CORE)
{
    // fprintf(stderr, "pack A starts \n");
    int mp  = mc / MR;
    int _mr = mc % MR;

    int i, j;
    int core = get_core_id();
    buffer += kc*MR*core;
    A += MR*incRowA*core;
    for (i=core; i<mp; i+=MATRIX_NUM_CORE) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR*MATRIX_NUM_CORE;
        A      += MR*incRowA*MATRIX_NUM_CORE;
    }
    
    if (_mr>0 && core == mp % MATRIX_NUM_CORE) {
        for (j=0; j<kc; ++j) {
            for (i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (i=_mr; i<MR; ++i) {
                buffer[i] = 0;
            }
            buffer += MR;
            A      += incColA;
        }
    }
    loki_tile_sync(MATRIX_NUM_CORE);
    // loki_sync_simple(MATRIX_NUM_CORE);
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, const Dtype *B, int incRowB, int incColB,
          Dtype *buffer)
{
    Dtype * limit = &B[k*incRowB];

    /*
    * the below is the original version
    */
    for (; B<limit; B+=incRowB) {
      asm volatile(
        "0: \n"
        "fetchr 1f \n"
        "ldw 0x0(%0) -> 10 \n"
        "ldw 0x4(%0) -> 11 \n"
        "ldw 0x8(%0) -> 12 \n"
        "ldw 0xc(%0) -> 13 \n"

        "stw r4, 0x0(%1) -> 1 \n"
        "stw r5, 0x4(%1) -> 1 \n"
        "stw r6, 0x8(%1) -> 1 \n"
        "stw.eop r7, 0xc(%1) -> 1 \n"

        "1: \n"
        
        : //"+&r"(B), "+&r"(buffer)
        : "r"(B), "r"(buffer)
        :
        );
        buffer += NR;
    }
}



//
//  Packing panels from B with padding if required
//
static void
pack_B(int kc, int nc, const Dtype *B, int incRowB, int incColB,
       Dtype *buffer, int MATRIX_NUM_CORE)
{
      int np  = nc / NR;
      int _nr = nc % NR;

      int i, j;

      int core = get_core_id();
      buffer += kc*NR*core;
      B += incColB*NR*core;
      for (j=core; j<np; j+=MATRIX_NUM_CORE) {
          pack_kxNR(kc, B, incRowB, incColB, buffer);
          buffer += kc*NR*MATRIX_NUM_CORE;
          B      += NR*incColB*MATRIX_NUM_CORE;
      }
      if (_nr>0 && core == np % MATRIX_NUM_CORE) {
          for (i=0; i<kc; ++i) {
              for (j=0; j<_nr; ++j) {
                  buffer[j] = B[j*incColB];
              }
              for (j=_nr; j<NR; ++j) {
                  buffer[j] = 0;
              }
              buffer += NR;
              B      += incRowB;
          }
      }
      loki_tile_sync(MATRIX_NUM_CORE);
      // loki_sync_simple(MATRIX_NUM_CORE);
}


static void
dgemm_micro_kernel( 
                   )
{

    asm volatile (
      // initialize 16 registers to hold intermediate values
      // "fetchr.eop 0f \n"
      // "0:"
      "scratchwri r9, 9 \n"
      "scratchwri r10, 10 \n"
      "scratchwri r8, 11 \n"
      "fetchr 0f \n"              // prefetching
      "and r10, r0, r10 \n"  // AB00
      "and r11, r0, r11 \n"  // AB10
      "and r12, r0, r12 \n"  // AB20
      "and r13, r0, r13 \n"  // AB30
      "and r14, r0, r14 \n"  // AB01
      "and r15, r0, r15 \n"  // AB11
      "and r16, r0, r16 \n"  // AB21
      "and r17, r0, r17 \n"  // AB31
      "and r18, r0, r18 \n"  // AB02
      "and r19, r0, r19 \n"  // AB12
      "and r20, r0, r20 \n"  // AB22
      "and r21, r0, r21 \n"  // AB32
      "and r22, r0, r22 \n"  // AB03
      "and r23, r0, r23 \n"  // AB13
      "and r24, r0, r24 \n"  // AB23
      "and r25, r0, r25 \n"  // AB33
      "scratchrdi r27, 1 \n"    // r27 now hold the value of A
      "addui r27, r27, -4 \n"
      "scratchrdi r28, 0 \n"  // r28 now hold the value of B
      "addui.eop r28, r28, -4 \n"
      /* check the loop end condition */
      "0: \n"
        // update the first column
        "addui r27, r27, 4 -> 10 \n"  // load the value of A[0]
        "addui r27, r27, 4 -> 11 \n"  // load the value of A[1]
        "addui r27, r27, 4 -> 12 \n"  // load the value of A[2]
        "addui r27, r27, 4 -> 13 \n"  // load the value of A[3]
        "or r8, r0, r4 \n"        // r8 permenantly hold the value of A[0]       
        "addui r28, r28, 4 -> 10 \n"  // load the value of B[0]
        "or r26, r0, r5 \n"    // r26 permenantly hold the value of A[1]
        "addui r28, r28, 4 -> 11 \n"  // load the value of B[1]
        "or r30, r0, r6 \n"    // r30 permenantly hold the value of A[2]
        "addui r28, r28, 4 -> 12 \n"  // load the value of B[2]
        "or r9, r0, r7 \n"         // r9 permenantly hold the value of A[3]
        "addui r28, r28, 4 -> 13 \n"  // load the value of B[3]

        // check loop condition for next packet fetching \n"
        "scratchrdi r31, 2 \n"        // load the value of limit
        "setlt.p r0, r28, r31 \n"     // compare if first limit has reached
        "psel.fetchr 0b, 1f\n"

        "or r29, r4, r0 \n"       // read the value of B[0] from channel
        "mullw r31, r29, r8 \n"
        "addu r10, r10, r31 \n"
        "mullw r31, r29, r26\n"
        "addu r11, r11, r31 \n"
        "mullw r31, r29, r30\n"
        "addu r12, r12, r31 \n"
        "mullw r31, r29, r9\n"  // r9 permenantly hold the value of A[3]
        "addu r13, r13, r31 \n"
        // update the 2nd column
        "or r29, r5, r0 \n"
        "mullw r31, r29, r8\n"
        "addu r14, r14, r31 \n"
        "mullw r31, r29, r26\n"
        "addu r15, r15, r31 \n"
        "mullw r31, r29, r30\n"
        "addu r16, r16, r31 \n"
        "mullw r31, r29, r9 \n"
        "addu r17, r17, r31 \n"
        // update the third column
        "or r29, r6, r0    \n"
        "mullw r31, r29, r8\n"
        "addu r18, r18, r31 \n"
        "mullw r31, r29, r26\n"
        "addu r19, r19, r31 \n"
        "mullw r31, r29, r30\n"
        "addu r20, r20, r31 \n"
        "mullw r31, r29, r9\n"
        "addu r21, r21, r31 \n"

        // update the fourth column
        "or r29, r7, r0 \n"
        "mullw r31, r29, r8\n"
        "addu r22, r22, r31 \n"
        "mullw r31, r29, r26\n"
        "addu r23, r23, r31 \n"
        "mullw r31, r29, r30\n"
        "addu r24, r24, r31 \n"
        "mullw r31, r29, r9\n"
        "addu.eop r25, r25, r31 \n"


      
      "1: \n"
      // "srai r10, r10, 0x0a \n"
      // "srai r11, r11, 0x0a \n"
      // "srai r12, r12, 0x0a \n"
      // "srai r13, r13, 0x0a \n"
      // "srai r14, r14, 0x0a \n"
      // "srai r15, r15, 0x0a \n"
      // "srai r16, r16, 0x0a \n"
      // "srai r17, r17, 0x0a \n"
      // "srai r18, r18, 0x0a \n"
      // "srai r19, r19, 0x0a \n"
      // "srai r20, r20, 0x0a \n"
      // "srai r21, r21, 0x0a \n"
      // "srai r22, r22, 0x0a \n"
      // "srai r23, r23, 0x0a \n"
      // "srai r24, r24, 0x0a \n"
      // "srai r25, r25, 0x0a \n"

      "scratchrdi r9, 9 \n"        // restore frame pointer
      "scratchrdi r28, 3 \n"       // r28 stores the address of C
      "scratchrdi r30, 5 \n"       // r30 incRowC

      "ldadd r10, r28, 0x00 -> 10 \n"
      "ldadd r14, r28, 0x04 -> 11 \n"
      "ldadd r18, r28, 0x08 -> 12 \n"
      "ldadd r22, r28, 0x0c -> 13 \n"

      "addu r28, r28, r30 \n"
      "or r0, r4, r5 \n" 
      "or r0, r6, r7 \n"
      "ldadd r11, r28, 0x00 -> 10 \n"
      "ldadd r15, r28, 0x04 -> 11 \n"
      "ldadd r23, r28, 0x0c -> 12 \n"
      "ldadd r19, r28, 0x08 -> 13 \n"    

      "addu r28, r28, r30 \n"
      "or r0, r4, r5 \n" 
      "or r0, r6, r7 \n"
      "ldadd r12, r28, 0x00 -> 10 \n"
      "ldadd r16, r28, 0x04 -> 11 \n"
      "ldadd r20, r28, 0x08 -> 12 \n"
      "ldadd r24, r28, 0x0c -> 13 \n"


      "addu r28, r28, r30 \n"

      "or r0, r4, r5 \n" 
      "or r0, r6, r7 \n"
      "fetchr 2f \n"
      "ldadd r13, r28, 0x00 -> 10 \n"
      "ldadd r17, r28, 0x04 -> 11 \n"
      "ldadd r21, r28, 0x08 -> 12 \n"
      "ldadd r25, r28, 0x0c -> 13 \n"
      
      // "or r0, r4, r5 \n" 
      // "or r0, r6, r7 \n"
      // "or r0, r4, r5 \n" 
      // "or r0, r6, r7 \n"
      // "or r0, r4, r5 \n" 
      // "or r0, r6, r7 \n"
      "scratchrdi r10, 10 \n"        // restore frame pointer
      "scratchrdi r8, 11 \n"
      "or r0, r4, r5 \n"
      "or.eop r0, r6, r7 \n"


      "2:"      
      :
      :
      : 
        "r11", "r12", "r13",
        "r14", "r15", "r16", "r17",
        "r18", "r19", "r20", "r21",
        "r22", "r23", "r24", "r25",
        
        "r27", "r28", "r29" ,"r31",

        "r26", "r30"
    );
}



//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   Dtype  *C,
                   Dtype  *B,
                   Dtype  *A,
                   int     incRowC,
                   int MATRIX_NUM_CORE
                   )
{
    uint core = get_core_id();
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int i, j;

    scratchpad_write(5, incRowC*4);   // integer of size 4 byte
    for (j=core; j<np; j+=MATRIX_NUM_CORE) {
        scratchpad_write(0, &B[j*kc*NR]);     // index 0 stores the address of B
        scratchpad_write(2, &B[(j+1)*kc*NR] - 4);
        for (i=0; i<mp; i++) {
            scratchpad_write(1, &A[i*kc*MR]);     // index 1 stores the address of A
            scratchpad_write(3, &C[i*MR*incRowC+j*NR]);     // index 3 stores the address of C
            dgemm_micro_kernel();
        }
    }
}

typedef struct global_data_ {
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
} global_data;


//
//  Compute C <- beta*C + A*B
//
void
dgemm_nn_worker(
         const void* data1)
{
    global_data* data = (global_data*)data1;
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
    int MATRIX_NUM_CORE = data->cores;
    /* set 4 channels so 4 input values can be read simultaneously
     * register 2 and 3 are avoided as they are used in synchrounization purpose 
     * channels below 10 are avoided as they might be used in libloki
    */
    int core = get_core_id();
    int addr = loki_mem_address(0, core, CH_REGISTER_4, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(10, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_5, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(11, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_6, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(12, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_7, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(13, addr);


    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;


    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;
        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;

              /* B values are packed in such a way that 
               * consecutive read are from contiguous
               * memory space
               */
              pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], 
                   incRowB, incColB,
                   _B, MATRIX_NUM_CORE);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;
                
                  /* A values are packed in such a way that 
                   * consecutive read are from contiguous
                   * memory space
                   */
                    pack_A(mc, kc,
                           &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                           _A, MATRIX_NUM_CORE);
                    
                    dgemm_macro_kernel(mc, nc, kc,
                                     &C[i*MC*incRowC+j*NC],
                                     _B,
                                     _A,
                                     incRowC, MATRIX_NUM_CORE);
            }
        }
    }
    loki_tile_sync(MATRIX_NUM_CORE);
    // loki_sync_simple(MATRIX_NUM_CORE);
}

void
dgemm_nn(int            m,
         int            n,
         int            k,
         const Dtype   *A,
         int            incRowA,
         const Dtype   *B,
         int            incRowB,
         Dtype         *C,
         int            incRowC
         // int MATRIX_NUM_CORE
         ){
                  int MATRIX_NUM_CORE = CONV_NUM_CORE;
                  loki_init_default(MATRIX_NUM_CORE, 0);
                  global_data* data = malloc(sizeof(global_data));
                  data->m = m;
                  data->n = n;
                  data->k = k;
                  data->C = C;
                  data->B = B;
                  data->A = A;
                  data->incRowA = incRowA;
                  data->incColA = 1;
                  data->incRowB = incRowB;
                  data->incColB = 1;
                  data->incRowC = incRowC;
                  data->incColC = 1;
                  data->cores = MATRIX_NUM_CORE;
                  
                  distributed_func* config = malloc(sizeof(distributed_func));
                  config->cores = MATRIX_NUM_CORE;
                  // set the function to be macro_kernel
                  config->func = &dgemm_nn_worker;
                  config->data = data;
                  config->data_size = sizeof(global_data);
                  loki_execute(config);
}

#endif