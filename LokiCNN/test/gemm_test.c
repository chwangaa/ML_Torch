#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
// #include "gemm_block_multicore_dataflow_parallelize_micro.h"
// #include "gemm_block_multicore_loopunroll_4.h"
// #include "gemm_block_multicore_dataflow_parallelize_micro.h"
// #include "data_structure.h"
// #include "math_functions.h"
// #include "setting.h"
// #include "../src/gemm_loki_multitile.h"
// #include "../src/gemm_loki_macro_p.h"
#include "../src/gemm_loki.h"
// #include "../src/gemm_loki_limited_stable.h"
// #include "gemm_blocks.h"
// #include "gemm_block_simple_integer.h"

// #include "gemm_block_simple_integer_full_asm.h"
// #include "util.h"
#include <loki/lokilib.h>
#include <fix8_loki.h>
/*
  test for when both conv layer is parallized by MultiCore
*/
// const int M = 128;
// const int N = 128;
// const int K = 128;

void* aligned_malloc(size_t size, size_t alignment) {
  void* pointer = malloc(size+alignment);
  void* aligned_pointer = pointer + (alignment - ((long)pointer % alignment));
  
  return aligned_pointer;
}


int test_gemm1(int M, int N, int K){
    int M1_incRow = K + (8 - (K % 8)) % 8;
    int M2_incRow = N + (8 - (N % 8)) % 8;
    int M3_incRow = N + (8 - (N % 8)) % 8;
    fprintf(stderr, "M1 has dimension %d x %d \n", M, M1_incRow);
    fprintf(stderr, "M2 has dimension %d x %d \n", K, M2_incRow);
    fprintf(stderr, "M3 has dimension %d x %d \n", M, M3_incRow);

    int* M1 = (int*)aligned_malloc(sizeof(int)*M*M1_incRow, 32);
    int* M2 = (int*)aligned_malloc(sizeof(int)*K*M2_incRow, 32);
    int* M3 = (int*)aligned_malloc(sizeof(int)*M*M3_incRow, 32);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            M1[i*M1_incRow+j] = read_from_int(1);
        }
    }
    
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            M2[i*M2_incRow+j] = read_from_int(1);
        }
    }
    
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            M3[i*M3_incRow+j] = 0;
        }
    }
    /* comment out this block if only using single core */
    for (int bank = 0; bank < 8; bank++) {
        int* address = (int*)(bank * 0x20);
        loki_channel_flush_all_lines(1, address);
        loki_channel_invalidate_all_lines(1, address);
    }

    unsigned long cycle_count = get_cycle_count();


    unsigned long instr_count = get_instruction_count();

    dgemm_nn(M, N, K,
            M1, M1_incRow,
            M2, M2_incRow,
            M3, M3_incRow);

    cycle_count = get_cycle_count() - cycle_count;
    instr_count = get_instruction_count() - instr_count;
    int error = 0;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(M3[i*M3_incRow+j] != read_from_int(M)){
                error++;
            }
        }
    }
    printf("M,  N,  K,  cycle,  instr,  #error \n");
    printf("%d, %d, %d, %d, %d, %d \n", M, N, K, cycle_count, instr_count, error);
    if(error){
        printf("TEST1 FAILS \n");
    }
    else{
        printf("TEST1 PASSES \n");
    }
}


int main(int argc, char** argv) {
  int M, N, K;
  if (argc < 4) {
    fprintf(stderr, "Usage: please give the value of M, N, K\n");
    return 2;
  }
  else{
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }
  test_gemm1(M, N, K);
  // test_gemm2(M, N, K);
}
