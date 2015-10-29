#include <loki/lokilib.h>
#include <loki/scratchpad.h>
#include "setting.h"

#define MC  32
#define KC  64
#define NC  128

#define MR  4
#define NR  4
#define MATRIX_NUM_CORE 8
// typedef int Dtype;
//
//  Local buffers for storing panels from A, B and C
//
static Dtype _A[MC*KC] __attribute__((aligned(0x1000)));
static Dtype _B[KC*NC] __attribute__((aligned(0x1000)));
// static Dtype _C[MR*NR];

void loki_sync_simple(int cores){
  if (cores <= 1)
    return;

  uint core = get_core_id();

  // (after setting up a connection).
  if (core > 0) {
    loki_receive_token(3);
  } 
  else {
    // All core 0s then synchronise between tiles using the same process.
      int bitmask = all_cores_except_0(cores);
      int address = loki_mcast_address(bitmask, 3, false);
      set_channel_map(3, address);
      loki_send_token(3);
  }
}
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
    /*
    * the below is an attempt to load in parallel, only work without loki_sync somehow
    */
    // asm volatile(
    //   "0: "
    //   "ldw 0x0(%0) -> 10 \n"
    //   "addu r28, %0, %3 \n"
    //   "ldw 0x0(r28) -> 11 \n"
    //   "addu r28, r28, %3 \n"
    //   "ldw 0x0(r28) -> 12 \n"
    //   "addu r28, r28, %3 \n"
    //   "ldw 0x0(r28) -> 13 \n"

    //   "setlt.p r0, %1, %2 \n"     // compare if first limit has reached
    //   "addui %0, %0, 0x04 \n"
    //   "psel.fetchr 0b, 1f\n" //? potential source of packet miss ?

    //   "stw r4, 0x0(%1) -> 1 \n"
    //   "stw r5, 0x4(%1) -> 1 \n"
    //   "stw r6, 0x8(%1) -> 1 \n"
    //   "stw r7, 0xc(%1) -> 1 \n"
    //   "addui.eop %1, %1, 0x10 \n"     // increment buffer
    //   "1: \n"
    //   : "+&r"(A), "+&r"(buffer)
    //   : "r"(limit), "r"(incRowA*4)
    //   : "r28"
    //   );  
}

//
//  Packing panels from A with padding if required
//
static void
pack_A(int mc, int kc, const Dtype *A, int incRowA, int incColA,
       Dtype *buffer)
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
    loki_sync_simple(MATRIX_NUM_CORE);
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, const Dtype *B, int incRowB, int incColB,
          Dtype *buffer)
{
    Dtype * limit = &B[k*incRowB];
    int _k = (k / 2 ) * 2;
    Dtype * _limit = &B[_k*incRowB];
    /*
    * the below is the original version
    */
    for (; B<_limit; B+=2*incRowB) {
        // for (int j=0; j<NR; ++j) {
        //     buffer[j] = B[j*incColB];
        // }
      /*
       * can fetch a whole cache line directly to the core
      */
      asm volatile(
        // "fetchr.eop 0f\n"
        "0: \n"
        // "sendconfig %0, 0xf -> 1 \n"
        "sendconfig %1, 0xf -> 1 \n"
        "fetchr 1f \n"
        "ldw 0x0(%0) -> 10 \n"
        "ldw 0x4(%0) -> 11 \n"
        "ldw 0x8(%0) -> 12 \n"
        "ldw 0xc(%0) -> 13 \n"

        "stw r4, 0x0(%1) -> 1 \n"
        "stw r5, 0x4(%1) -> 1 \n"
        "stw r6, 0x8(%1) -> 1 \n"
        "stw r7, 0xc(%1) -> 1 \n"

        "addu %0, %0, %2 \n"
        "ldw 0x0(%0) -> 10 \n"
        "ldw 0x4(%0) -> 11 \n"
        "ldw 0x8(%0) -> 12 \n"
        "ldw 0xc(%0) -> 13 \n"

        "stw r4, 0x10(%1) -> 1 \n"
        "stw r5, 0x14(%1) -> 1 \n"
        "stw r6, 0x18(%1) -> 1 \n"
        "stw.eop r7, 0x1c(%1) -> 1 \n"

        "1: \n"
        : 
        : "r"(B), "r"(buffer), "r"(incRowB*4)
        :
        );
        buffer += 2*NR;
        // B      += incRowB;
    }
    /*
    *  the below is the optimized version
    */
    // asm volatile(
    //   "fetchr.eop 0f\n"
    //   "0: \n"
    //   "ldw 0x0(%0) -> 10 \n"
    //   "ldw 0x4(%0) -> 11 \n"
    //   "ldw 0x8(%0) -> 12 \n"
    //   "ldw 0xc(%0) -> 13 \n"

    //   "setlt.p r0, %0, %2 \n"
    //   "addu %0, %0, %3 \n"
    //   "psel.fetchr 0b, 1f\n"

    //   "stw r4, 0x0(%1) -> 1 \n"
    //   "stw r5, 0x4(%1) -> 1 \n"
    //   "stw r6, 0x8(%1) -> 1 \n"
    //   "stw r7, 0xc(%1) -> 1 \n"
    //   "addui.eop %1, %1, 0x10 \n"     // increment buffer
    //   "1: \n"
    //   : "+&r"(B), "+&r"(buffer)
    //   : "r"(limit), "r"(incRowB*4)
    //   :
    //   );
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
      loki_sync_simple(MATRIX_NUM_CORE);
}


static void
dgemm_micro_kernel( 
                   )
{

    asm volatile (
      // "fetchr.eop 0f \n"
      "0:"
      "fetchr 1f \n"
      "scratchrdi r27, 1 \n"    // r27 now hold the value of A
      "scratchwri.eop r9, 9 \n"
      
      "1:"
      "fetchr 2f \n"
      // initialize 16 registers to hold intermediate values
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
      "scratchrdi.eop r28, 0 \n"  // r28 now hold the value of B

      /* check the loop end condition */
      "2: \n"
        // update the first column
        "ldw 0x0(r27) -> 10 \n"   // load the value of A[0]
        "ldw 0x4(r27) -> 11 \n"    // load the value of A[1]
        "ldw 0x8(r27) -> 12 \n"    // load the value of A[2]
        "ldw 0xc(r27) -> 13 \n"    // load the value of A[3]
        "or r8, r0, r4 \n"        // r8 permenantly hold the value of A[0]       
        "ldw 0x0(r28) -> 10 \n"    // load the value of B[0]
        "or r26, r0, r5 \n"    // r26 permenantly hold the value of A[1]
        "ldw 0x4(r28) -> 11 \n"    // load the value of B[1]
        "or r30, r0, r6 \n"    // r30 permenantly hold the value of A[2]
        "ldw 0x8(r28) -> 12 \n"    // load the value of B[2]
        "or r9, r0, r7 \n"
        "ldw 0xc(r28) -> 13 \n"    // load the value of B[3]

        // check loop condition for next packet fetching \n"
        "scratchrdi r29, 2 \n"        // load the value of limit

        "addui r28, r28, 0x10 \n"     // sincrement B        
        "setlt.p r0, r28, r29 \n"     // compare if first limit has reached
        "addui r27, r27, 0x10 \n"     // increment A
        "psel.fetchr 2b, 3f\n" //? potential source of packet miss ?
        // "addu r29, r29, r29 \n"
        // "addu r29, r29, r29 \n"
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

      // "1: \n"      
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
      
      "3: \n"
      // "sendconfig r28, 0xf -> 1 \n"
      "scratchrdi r29, 4 \n"        // load the value of limit
      "setlt.p r0, r27, r29 \n"     // compare if all A has been iterated
      "scratchrdi r28, 3 \n"        // r28 stores the address of C
      "scratchrdi r30, 5 \n"        // r30 incRowC      
      "psel.fetchr 1b, 4f\n"        //? potential source of packet miss

      "ldadd r10, r28, 0x00 -> 10 \n"
      "ldadd r14, r28, 0x04 -> 11 \n"
      "ldadd r18, r28, 0x08 -> 12 \n"
      "ldadd r22, r28, 0x0c -> 13 \n"
      "addu r28, r28, r30 \n"
      "or r0, r4, r0 \n"
      "or r0, r5, r0 \n"
      "or r0, r6, r0 \n"
      "or r0, r7, r0 \n"


      "ldadd r11, r28, 0x00 -> 10 \n"
      "ldadd r15, r28, 0x04 -> 11 \n"
      "ldadd r23, r28, 0x0c -> 12 \n"
      "ldadd r19, r28, 0x08 -> 13 \n"
      "addu r28, r28, r30 \n"
      "or r0, r4, r0 \n"
      "or r0, r5, r0 \n"
      "or r0, r6, r0 \n"
      "or r0, r7, r0 \n"


      "ldadd r12, r28, 0x00 -> 10 \n"
      "ldadd r16, r28, 0x04 -> 11 \n"
      "ldadd r20, r28, 0x08 -> 12 \n"
      "ldadd r24, r28, 0x0c -> 13 \n"
      "addu r28, r28, r30 \n"
      "or r0, r4, r0 \n"
      "or r0, r5, r0 \n" 
      "or r0, r6, r0 \n"
      "or r0, r7, r0 \n"


      "ldadd r13, r28, 0x00 -> 10 \n"
      "ldadd r17, r28, 0x04 -> 11 \n"
      "ldadd r21, r28, 0x08 -> 12 \n"
      "ldadd r25, r28, 0x0c -> 13 \n"
      "addu r28, r28, r30 \n"
      "or r0, r4, r0 \n"
      "or r0, r5, r0 \n"
      "or r0, r6, r0 \n"
      "or r0, r7, r0 \n"


      "scratchwri.eop r28, 3 \n"

      "4:"
      // update the value of B
      "scratchrdi r28, 0 \n"      // load B
      "scratchrdi r27, 8 \n"      // load incrementB
      "scratchrdi r30, 10 \n"     // load limitB
      "addu r28, r28, r27 \n"     // increment B
      "setlt.p r0, r28, r30 \n"
      "scratchwri r28, 0 \n"
      "psel.fetchr 1b, 5f\n"        //? potential source of packet miss
      "scratchrdi r27, 11 \n"
      "addu r28, r28, r27 \n"
      "scratchwri r28, 2 \n"

      // update the value of C
      "scratchrdi r28, 6 \n"
      "scratchrdi r27, 7 \n"
      "addu r27, r27, r28 \n"
      "scratchwri r27, 7 \n"
      "scratchwri r27, 3 \n"
      // load the value of A
      "scratchrdi.eop r27, 1 \n"

      "5:"
      "fetchr 6f \n"
      "scratchrdi.eop r9, 9 \n"

      "6:"
      :
      :
      : 
        "r10", "r11", "r12", "r13",
        "r14", "r15", "r16", "r17",
        "r18", "r19", "r20", "r21",
        "r22", "r23", "r24", "r25",
        "r27", "r28", "r29" ,"r31",

        "r9", "r8", "r26", "r30"
    );
}



//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
dgemm_macro_kernel(const int     mc,
                   const int     nc,
                   const int     kc,
                   Dtype  *C,
                   Dtype  *B,
                   Dtype  *A,
                   const int     incRowC
                   // int     incColC
                   )
{
    uint core = get_core_id();
    const int mp = (mc+MR-1) / MR;
    const int np = (nc+NR-1) / NR;

    // int i, j;
    const Dtype* LIMIT = &B[np*kc*NR];
    B = &B[core*kc*NR];
    C = &C[core*NR];
    scratchpad_write(5, incRowC*4);   // integer of size 4 byte
    scratchpad_write(1, A);     // index 1 stores the address of A
    scratchpad_write(4, &A[mp*kc*MR]);
    scratchpad_write(3, C);
    scratchpad_write(7, C);
    scratchpad_write(6, MATRIX_NUM_CORE*NR*4);
    scratchpad_write(8, MATRIX_NUM_CORE*kc*NR*4);
    scratchpad_write(10, LIMIT);
    scratchpad_write(11, kc*NR*4);
    // for (; B < LIMIT; B += MATRIX_NUM_CORE*kc*NR) {
    scratchpad_write(0, B);     // index 0 stores the address of B
    scratchpad_write(2, B + kc*NR);
        // scratchpad_write(3, C);     // index 3 stores the address of AB
    dgemm_micro_kernel();
    // }
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

              pack_B(kc, nc,
                   &B[l*KC*incRowB+j*NC*incColB], 
                   incRowB, incColB,
                   _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;
                
                  // core 0 transpose and pack the value
                  // if(core == 0){
                    pack_A(mc, kc,
                           &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                           _A);
                  // }
                  // loki_sync_simple(MATRIX_NUM_CORE);
                  dgemm_macro_kernel(mc, nc, kc,
                                     &C[i*MC*incRowC+j*NC],
                                     _B,
                                     _A,
                                     incRowC);
            }
        }
    }
    loki_sync_simple(MATRIX_NUM_CORE);
}

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
         ){

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