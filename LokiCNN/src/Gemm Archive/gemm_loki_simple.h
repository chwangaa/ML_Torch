#include "setting.h"
#include "data_structure.h"
#include <loki/lokilib.h>

#define MC  64
#define KC  64
#define NC  128

#define MR  4
#define NR  4

//
//  Local buffers for storing panels from A, B and C
//
static Dtype _A[MC*KC];
static Dtype _B[KC*NC];
static Dtype _C[MR*NR];

//
//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk(int k, const Dtype *A, int incRowA, int incColA,
          Dtype *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A(int mc, int kc, const Dtype *A, int incRowA, int incColA,
       Dtype *buffer)
{
    int mp  = mc / MR;
    int _mr = mc % MR;

    int i, j;

    for (i=0; i<mp; ++i) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    if (_mr>0) {
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
}


__attribute__ ((noinline)) static void
dgemm_micro_kernel_set(int kc,
                   const Dtype *A, const Dtype *B,
                   Dtype *C, int incRowC, int incColC)
{

    Dtype* limit = &B[kc*NR];
    
    scratchpad_write(0, B);     // index 0 stores the address of B
    scratchpad_write(1, A);     // index 1 stores the address of A
    scratchpad_write(2, limit); // index 2 stores the upper bound

    scratchpad_write(3, incRowC*4); // index 3 stores the row offset
    scratchpad_write(4, incColC*4); // index 4 stores the col offset

    asm volatile (
      // initialize 16 registers to hold intermediate values
      "and r10, r0, r10 \n"  // AB00
      "and r11, r0, r11 \n"  // AB10
      "and r12, r0, r12 \n"  // AB20
      "and r13, r0, r13 \n"  // AB30
      
      "and r14, r0, r14 \n"  // AB01
      "and r15, r0, r15 \n"  // AB11
      "and r16, r0, r16 \n"  // AB21
      "and r17, r0, r17 \n"  // AB31
      "fetchr loop_condition2 \n"              // prefetching      
      "and r18, r0, r18 \n"  // AB02
      "and r19, r0, r19 \n"  // AB12
      "and r20, r0, r20 \n"  // AB22
      "and r21, r0, r21 \n"  // AB32
      
      "and r22, r0, r22 \n"  // AB03
      "and r23, r0, r23 \n"  // AB13
      "and r24, r0, r24 \n"  // AB23
      "and.eop r25, r0, r25 \n"  // AB33


      /* check the loop end condition */
      "loop_condition2: \n"
      "scratchrdi r28, 0 \n"  // r28 now hold the value of B
      "scratchrdi r29, 2 \n"  // load the value of limit
      "setlt.p r0, r28, r29 \n" // compare if limit has reached
      "psel.fetchr.eop 1f, 0f\n" //? potential source of packet miss ?

      // exit the loop
      "0: \n"
      "fetchr.eop exit2 \n"

      // increment A, B, and then proceed
      "1: \n"
      "ldw 0x0(r28) -> 1 \n"    // load the value of B[0][0]
      "addui r29, r28, 0x10 \n" // B+=NR
      "scratchwri r29, 0 \n"    // store the value back to scratchpad
      "or r29, r2, r0 \n"       // read the value of B[0][0] from channel
      "scratchrdi r27, 1 \n"    // r27 now hold the value of A
      "addui r31, r27, 0x10 \n" // A+=MR
      "fetchr 1f \n"            // pre-fetch the next instruction packet
      "scratchwri r31, 1 \n"    // store the value back to scratchpad
      
      "ldw.eop 0x0(r27) -> 1 \n"  // load the value of A[0][0]


        // execute the loop body
        
        // update the first column              
        "1: \n"
        "mullw r31, r29, r2 \n"
        "ldw 0x4(r27) -> 2 \n"
        "addu r10, r10, r31 \n"
        
        "mullw r31, r29, r3\n"
        "ldw 0x8(r27) -> 2 \n"
        "addu r11, r11, r31 \n"

        "mullw r31, r29, r3\n"
        "ldw 0xc(r27) -> 2 \n"
        "addu r12, r12, r31 \n"

        "mullw r31, r29, r3\n"
        "fetchr 2f \n"
        "ldw 0x4(r28) -> 1 \n"
        "ldw 0x0(r27) -> 2 \n"
        "addu.eop r13, r13, r31 \n"

        // update the 2nd column
        "2: \n"
        "or r29, r2, r0 \n"
        
        "mullw r31, r29, r3\n"
        "ldw 0x4(r27) -> 2 \n"        
        "addu r14, r14, r31 \n"
        
        "mullw r31, r29, r3\n"
        "ldw 0x8(r27) -> 2 \n"
        "addu r15, r15, r31 \n"

        "mullw r31, r29, r3\n"
        "ldw 0xc(r27) -> 2 \n"
        "addu r16, r16, r31 \n"

        "mullw r31, r29, r3\n"
        "fetchr 3f \n"
        "ldw 0x8(r28) -> 1 \n"
        "ldw 0x0(r27) -> 2 \n"
        "addu.eop r17, r17, r31 \n"

        // update the third column
        "3: \n"
        "or r29, r2, r0 \n"
        
        "mullw r31, r29, r3\n"
        "ldw 0x4(r27) -> 2 \n"
        "addu r18, r18, r31 \n"
        
        "mullw r31, r29, r3\n"
        "ldw 0x8(r27) -> 2 \n"
        "addu r19, r19, r31 \n"

        "mullw r31, r29, r3\n"
        "ldw 0xc(r27) -> 2 \n"
        "addu r20, r20, r31 \n"

        "mullw r31, r29, r3\n"
        "fetchr 4f \n"
        "ldw 0xc(r28) -> 1 \n"
        "ldw 0x0(r27) -> 2 \n"
        "addu.eop r21, r21, r31 \n"

        // update the fourth column
        "4: \n"
        "or r29, r2, r0 \n"        

        "mullw r31, r29, r3\n"
        "ldw 0x4(r27) -> 2 \n"
        "addu r22, r22, r31 \n"
        
        "mullw r31, r29, r3\n"
        "ldw 0x8(r27) -> 2 \n"
        "addu r23, r23, r31 \n"

        "mullw r31, r29, r3\n"
        "ldw 0xc(r27) -> 2 \n"
        "addu r24, r24, r31 \n"

        "mullw r31, r29, r3\n"
        "fetchr loop_condition2 \n"
        "addu.eop r25, r25, r31 \n"
      
      "exit2: \n"
      "or r27, %0, r0 \n"     // r27 hold the value of C
      "scratchrdi r28, 4 \n"  // r28 incCol
      "scratchrdi r29, 3 \n"  // r29 incRow
      "stw r10, 0x00(r27) ->1\n"
      "stw r11, 0x04(r27) ->2\n"
      "stw r12, 0x08(r27) ->1\n"
      "stw r13, 0x0c(r27) ->2\n"
      
      "addu r27, r27, r28 \n"
      "stw r14, 0x00(r27) ->1\n"
      "stw r15, 0x04(r27) ->2\n"
      "stw r16, 0x08(r27) ->1\n"
      "stw r17, 0x0c(r27) ->2\n"
      "fetchr 0f \n"

      "addu r27, r27, r28 \n"      
      "stw r18, 0x00(r27) ->1\n"
      "stw r19, 0x04(r27) ->2\n"
      "stw r20, 0x08(r27) ->1\n"
      "stw r21, 0x0c(r27) ->2\n"

      "addu r27, r27, r28 \n"
      "stw r22, 0x00(r27) ->1\n"
      "stw r23, 0x04(r27) ->2\n"
      "stw r24, 0x08(r27) ->1\n"
      "stw.eop r25, 0x0c(r27) ->2\n"
      "0:"      
      :
      : "r" (C)
      : 
        "r10", "r11", "r12", "r13",
        "r14", "r15", "r16", "r17",
        "r18", "r19", "r20", "r21",
        "r22", "r23", "r24", "r25",
        "r27", "r28", "r29" ,"r31"
    );
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, const Dtype *B, int incRowB, int incColB,
          Dtype *buffer)
{
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

//
//  Packing panels from B with padding if required
//
static void
pack_B(int kc, int nc, const Dtype *B, int incRowB, int incColB,
       Dtype *buffer)
{
    int np  = nc / NR;
    int _nr = nc % NR;

    int i, j;

    for (j=0; j<np; ++j) {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    if (_nr>0) {
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
}

//
//  Micro kernel for multiplying panels from A and B.
//
__attribute__ ((noinline)) static void
dgemm_micro_kernel(int kc,
                   const Dtype *A, const Dtype *B,
                   Dtype beta,
                   Dtype *C, int incRowC, int incColC)
{
    Dtype AB[MR*NR];

    int i, j, l;

//z
//  Compute AB = A*B
//
    for (l=0; l<MR*NR; ++l) {
        AB[l] = 0;
    }
    Dtype* limit = &B[kc*NR];
    for (; B < limit; A+=MR, B+=NR) {
            
            asm volatile (
              // update the first column

              "fetchr.eop 1f \n"
              "1: \n"
              "ldw 0x0(%2) -> 1 \n"
              "or r29, r2, r0 \n"
              
              "ldw 0x0(%1) -> 1 \n"
              "ldw 0x0(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x0(%0) -> 1\n"
              
              "ldw 0x4(%1) -> 1 \n"
              "ldw 0x4(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x4(%0) -> 1\n"

              "ldw 0x8(%1) -> 1 \n"
              "ldw 0x8(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x8(%0) -> 1\n"

              "ldw 0xc(%1) -> 1 \n"
              "ldw 0xc(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "fetchr 2f \n"
              "addu r31, r3, r31 \n"
              "stw.eop r31, 0xc(%0) -> 1\n"

              // update the 2nd column
              "2: \n"
              "ldw 0x4(%2) -> 1 \n"
              "or r29, r2, r0 \n"
              
              "ldw 0x0(%1) -> 1 \n"
              "ldw 0x10(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x10(%0) -> 1\n"
              
              "ldw 0x4(%1) -> 1 \n"
              "ldw 0x14(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x14(%0) -> 1\n"

              "ldw 0x8(%1) -> 1 \n"
              "ldw 0x18(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x18(%0) -> 1\n"

              "ldw 0xc(%1) -> 1 \n"
              "ldw 0x1c(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "fetchr 3f \n"
              "addu r31, r3, r31 \n"
              "stw.eop r31, 0x1c(%0) -> 1\n"
              "3: \n"

              // update the third column
              "3: \n"
              "ldw 0x8(%2) -> 1 \n"
              "or r29, r2, r0 \n"
              
              "ldw 0x0(%1) -> 1 \n"
              "ldw 0x20(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x20(%0) -> 1\n"
              
              "ldw 0x4(%1) -> 1 \n"
              "ldw 0x24(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x24(%0) -> 1\n"

              "ldw 0x8(%1) -> 1 \n"
              "ldw 0x28(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x28(%0) -> 1\n"

              "ldw 0xc(%1) -> 1 \n"
              "ldw 0x2c(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "fetchr 4f \n"
              "addu r31, r3, r31 \n"
              "stw.eop r31, 0x2c(%0) -> 1\n"
              "4: \n"

              // update the fourth column
              "4: \n"
              "ldw 0xc(%2) -> 1 \n"
              "or r29, r2, r0 \n"
              
              "ldw 0x0(%1) -> 1 \n"
              "ldw 0x30(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x30(%0) -> 1\n"
              
              "ldw 0x4(%1) -> 1 \n"
              "ldw 0x34(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x34(%0) -> 1\n"

              "ldw 0x8(%1) -> 1 \n"
              "ldw 0x38(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "addu r31, r3, r31 \n"
              "stw r31, 0x38(%0) -> 1\n"

              "ldw 0xc(%1) -> 1 \n"
              "ldw 0x3c(%0) -> 2 \n"
              "mullw r31, r29, r2\n"
              "fetchr 5f \n"
              "addu r31, r3, r31 \n"
              "stw.eop r31, 0x3c(%0) -> 1\n"
              "5: \n"

              :
              : "r" (&AB[0]), "r" (&A[i]), "r" (&B[0])
              : 
                // "r10", "r11",
                // "r12","r13","r14", "r15",
                // "r16","r17","r18","r19",
                // "r20", "r21","r22","r23",
                // "r24", "r25",
                "r29", "r30", "r31"
            );
        // }
        // A += MR;
        // B += NR;
    }


//
//  if beta is set, C = A*B
//
    if (beta==0.0) {
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = AB[i+j*MR];
            }
        }
    }
//
//  else, C += A*B
//
    else{
        for (j=0; j<NR; ++j) {
            for (i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*MR];
            }
        }
    }
}

//
//  Compute Y += alpha*X
//
static void
dgeaxpy(int           m,
        int           n,
        Dtype        alpha,
        const Dtype  *X,
        int           incRowX,
        int           incColX,
        Dtype        *Y,
        int           incRowY,
        int           incColY)
{
    int i, j;


    if (alpha!=1.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha * X[i*incRowX+j*incColX];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
static void
dgescal(int     m,
        int     n,
        Dtype  alpha,
        Dtype  *X,
        int     incRowX,
        int     incColX)
{
    int i, j;

    if (alpha!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = X[i*incRowX+j*incColX] * alpha;
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0;
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
__attribute__ ((noinline)) static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   Dtype  *C,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR],
                                   1,
                                   &C[i*MR*incRowC+j*NR*incColC],
                                   incRowC, incColC);
            } else {
                dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR],
                                   0,
                                   _C, 1, MR);
                dgeaxpy(mr, nr, 1.0, _C, 1, MR,
                        &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}

//
//  Compute C <- beta*C + A*B
//
void
dgemm_nn(int            m,
         int            n,
         int            k,
         const Dtype   *A,
         int            incRowA,
         // int            incColA,
         const Dtype   *B,
         int            incRowB,
         // int            incColB,
         // Dtype         beta,
         Dtype         *C,
         int            incRowC
         // int            incColC
         )
{
    int incColA = 1;
    int incColB = 1;
    int incColC = 1;
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    Dtype _beta;

  int core = get_core_id();
  int addr = loki_mem_address(0, core, CH_REGISTER_3, GROUPSIZE_8, false, false, true, true);  
  set_channel_map(2, addr);

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;

            pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                   _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                       _A);

                dgemm_macro_kernel(mc, nc, kc,
                                   &C[i*MC*incRowC+j*NC*incColC],
                                   incRowC, incColC);
            }
        }
    }
}