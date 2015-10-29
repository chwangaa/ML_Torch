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

#include "gemv_loki.h"
// #include "gemm_blocks.h"
// #include "gemm_block_simple_integer.h"

// #include "gemm_block_simple_integer_full_asm.h"
// #include "util.h"
#include <loki/lokilib.h>
/*
  test for when both conv layer is parallized by MultiCore
*/

const int M = 128;
const int N = 1;
const int K = 128;

void* aligned_malloc(size_t size, size_t alignment) {
  void* pointer = malloc(size+alignment);
  void* aligned_pointer = pointer + (alignment - ((long)pointer % alignment));
  
  return aligned_pointer;
}


int main(){
 //    const int M = 128;
	// const int MATRIX_SIZE = 16;

	int* M1 = (int*)aligned_malloc(sizeof(int)*M*K, 32);
	int* M2 = (int*)aligned_malloc(sizeof(int)*N*K, 32);
	int* M3 = (int*)aligned_malloc(sizeof(int)*M*N, 32);

    // int* M1 = (int*)malloc(sizeof(int)*M*K);
    // int* M2 = (int*)malloc(sizeof(int)*N*K);
    // int* M3 = (int*)malloc(sizeof(int)*M*N);

    // for(int i = 0; i < MATRIX_SIZE; i++){
    //     M1[i] = read_from_int(_M1[i]);
    //     M2[i] = read_from_int(_M2[i]);
    //     M3[i] = read_from_int(_M3[i]);
    // }
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            M1[i*K+j] = 1;
            // M1[i*K+j] = read_from_int(1);
        }
    }
    
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            M2[i*N+j] = 1;
            // M2[i*N+j] = read_from_int(1);
        }
    }
    
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            M3[i*N+j] = 0;
            // M3[i*N+j] = read_from_int(0);
        }
    }

    // for (int bank = 0; bank < 8; bank++) {
    //     int* address = (int*)(bank * 0x20);
    //     loki_channel_flush_all_lines(1, address);
    //     loki_channel_invalidate_all_lines(1, address);
    // }
	// for(int i = 0; i < MATRIX_SIZE; i++){
	// 	M1[i] = readDouble(1);
	// 	M2[i] = readDouble(1);
	// 	M3[i] = readDouble(0.5);
	// }
    unsigned long cycle_count = get_cycle_count();


	unsigned long instr_count = get_instruction_count();
        // dgemm_nn(M, N, K,
        //         M1, K,1,
        //          M2, N,1,
        //         M3, N,1);

    dgemv_nn(M, K,
            M1, K,
            M2,
            M3);


    cycle_count = get_cycle_count() - cycle_count;
    instr_count = get_instruction_count() - instr_count;
    fprintf(stderr, "takes %lu cycle to complete \n", cycle_count);
    fprintf(stderr, "takes %lu instructions \n", instr_count);


    for(int i = 0; i < M; i++){
    // 	// fprintf(stderr, "%d \n", fix16_to_int(M3[i]));
    	for(int j = 0; j < N; j++){
            if(M3[i*N+j] != M)
    		  fprintf(stderr, "%d  at %d, %d \n", M3[i*N+j], i, j);
            // fprintf(stderr, "%d ", fix8_to_int(M3[i*N+j]));
            // fprintf(stderr, "%d ", M3[i*N+j]);
        }
    	// fprintf(stderr, "\n\n");
    }
}