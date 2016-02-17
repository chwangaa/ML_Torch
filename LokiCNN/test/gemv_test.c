#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>

// #include "data_structure.h"
#include "math_functions.h"

#include "../src/gemv_loki.h"
#include <loki/lokilib.h>
/*
  test for when both conv layer is parallized by MultiCore
*/

void* aligned_malloc(size_t size, size_t alignment) {
  void* pointer = malloc(size+alignment);
  void* aligned_pointer = pointer + (alignment - ((long)pointer % alignment));
  
  return aligned_pointer;
}

/*
 * Muliply a MxK matrix by a Kx1 vector.
 *
 * All elements are 1, so the result should be an 1xM vector where every value
 * is K.
 */
int main(int argc, char** argv){
  int M, N, K;
  if (argc < 3) {
    fprintf(stderr, "Usage: please give the value of M and K\n");
    return 2;
  }
  else{
    M = atoi(argv[1]);
    N = 1;
    K = atoi(argv[2]);
  }

	Dtype* M1 = (Dtype*)aligned_malloc(sizeof(Dtype)*M*K, 32);
	Dtype* M2 = (Dtype*)aligned_malloc(sizeof(Dtype)*N*K, 32);
	Dtype* M3 = (Dtype*)aligned_malloc(sizeof(Dtype)*M*N, 32);

  for(int i = 0; i < M; i++){
      for(int j = 0; j < K; j++){
          M1[i*K+j] = ONE;
      }
  }
  
  for(int i = 0; i < K; i++){
      for(int j = 0; j < N; j++){
          M2[i*N+j] = ONE;
      }
  }
  
  for(int i = 0; i < M; i++){
      for(int j = 0; j < N; j++){
          M3[i*N+j] = 0;
      }
  }
  
  unsigned long cycle_count = get_cycle_count();
	unsigned long instr_count = get_instruction_count();

  dgemv_nn(M, K,
          M1, K,
          M2,
          M3);

  cycle_count = get_cycle_count() - cycle_count;
  instr_count = get_instruction_count() - instr_count;

  int errors = 0;

  for(int i = 0; i < M; i++){
  	for(int j = 0; j < N; j++){
  	  int val = to_int(M3[i*N + j]);
      if(val != K) {
        errors++;
		    fprintf(stderr, "%d  at %d, %d \n", val, i, j);
		  }
    }
  }
  
  printf("M,\tN,\tK,\tcycle,\tinstr,\t#errors \n");
  printf("%d,\t%d,\t%d,\t%lu,\t%lu,\t%d \n", M, N, K, cycle_count, instr_count, errors);
}
