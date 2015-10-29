// Strassen implementation ----------------------------------------------------
#ifndef STRASSEN_M_H
#define STRASSEN_M_H

#define CACHE_SIZE 256*1024
#define THREADS 7

#include <assert.h>
#include <math.h>
#include <pthread.h>

#include "data_structure_2.h"

//multi-threaded implementation of the strassen algorithm.
//less memory efficient than the single thread version
//(needs separate memory for each multiplication)

void* strassen(void* args);

static void inline adeqar(arr2_t* a, arr2_t* b) {
  assert(a->sx >= b->sx && a->sy >= b->sy);
  int index = 0;
  for(int y=0; y<b->sy; y++)
    for(int x=0; x<b->sx; x++)
      set_arr2(a,x,y,get_arr2(a,x,y) + b->data[index++]);
}

static void inline resize(arr2_t* A,int x,int y) {
  A->sx = x;
  A->sy = y;
}

static void inline sueqar(arr2_t* a, arr2_t* b) {
  assert(a->sx >= b->sx && a->sy >= b->sy);
  int index = 0;
  for(int y=0; y<b->sy; y++)
    for(int x=0; x<b->sx; x++)
      set_arr2(a,x,y,get_arr2(a,x,y) - b->data[index++]);
}

static void inline negate(arr2_t* arr) {
  int cap = arr->sx * arr->sy;
  for(int i=0; i<cap; i++)
    arr->data[i] = -arr->data[i];
}

pthread_t t[THREADS-1];

int top;

typedef struct thread_data {
  int thread_number;
  arr2_t** arrs;
} thread_data_t;

void* thread_init(void* args) {
  thread_data_t* data = args;
  for(int i=data->thread_number; i<7; i+=THREADS) {
    // printf("%d\n",i);
    strassen(data->arrs + i*3);
  }
}

void* strassen(void* args) {

  int spawn = 0;

  if(top) {
    spawn = 1;
    top = 0;
  }

  arr2_t** temp = (arr2_t**)args;
  arr2_t* A = temp[1];
  arr2_t* B = temp[2];
  arr2_t* result = temp[0];

  printf("%d,%d,%d,%d\n",A->sx,A->sy,B->sx,B->sy);
  int m = A->sy;
  int n = A->sx;
  assert(n == B->sy);
  int k = B->sx;

  assert(result->sx == k && result->sy == m);

  int size = fmax(m,n);
  size = fmax(size,k);

  int area = sizeof(storage_t)*n*(k+m);

  if(area<=CACHE_SIZE) {
    for(int a=0; a<m; a++) {
      for(int b=0; b<k; b++) {
        storage_t accum = 0;
        for(int c=0; c<n; c++) {
          accum += get_arr2(A,c,a) * get_arr2(B,b,c);
        }
        set_arr2(result,b,a,accum);
      }
    }
    return NULL;
  }

  if(size == 1) {
    set_arr2(result,0,0, get_arr2(A,0,0) * get_arr2(B,0,0));
    return NULL;
  }
 
  int cap = (size+1)>>1;

  arr2_t *D[8];     //store split matricies
  arr2_t *C[21];    //used for the computations

  if(m>1 && n>1 && k>1) {

    //prepare data

    printf("7\n");

    int mcap = (m+1)/2;
    int ncap = (n+1)/2;
    int kcap = (k+1)/2;
    D[0] = make_arr2(ncap,mcap);
    fill(D[0],A,0,0,ncap,mcap);           //[A1, A2][B1, B2]= [R1, R2]
    D[1] = make_arr2(n-ncap,mcap);        //[A3, A4][B3, B4]  [R3, R4]
    fill(D[1],A,ncap,0,n,mcap);           //M1= (A1 +A4)(B1 +B4)
    D[2] = make_arr2(ncap,m-mcap);        //M2= (A3 +A4)(B1)
    fill(D[2],A,0,mcap,ncap,m);           //M3= (A1)(B2 -B4)
    D[3] = make_arr2(n-ncap,m-mcap);      //M4= (A4)(B3 -B1)
    fill(D[3],A,ncap,mcap,n,m);           //M5= (A1 +A2)(B4)
    D[4] = make_arr2(kcap,ncap);          //M6= (A3 -A1)(B1 +B2)
    fill(D[4],B,0,0,kcap,ncap);           //M7= (A2 -A4)(B3 +B4)
    D[5] = make_arr2(k-kcap,ncap);        //R1= M1 +M4 -M5 +M7
    fill(D[5],B,kcap,0,k,ncap);           //R2= M3 +M5
    D[6] = make_arr2(kcap,n-ncap);        //R3= M2 +M4
    fill(D[6],B,0,ncap,kcap,n);           //R4= M1 -M2 +M3 +M6
    D[7] = make_arr2(k-kcap,n-ncap);
    fill(D[7],B,kcap,ncap,k,n);

    C[0] = make_arr2(kcap,mcap);
    C[1] = make_arr2(ncap,mcap);
    copy_arr2(C[1],D[0]);
    adeqar(C[1],D[3]);
    C[2] = make_arr2(kcap,ncap);
    copy_arr2(C[2],D[4]);
    adeqar(C[2],D[7]);

    C[3] = make_arr2(kcap,m-mcap);
    C[4] = make_arr2(ncap,m-mcap);
    copy_arr2(C[4],D[2]);
    adeqar(C[4],D[3]);
    C[5] = make_arr2(kcap,ncap);
    copy_arr2(C[5],D[4]);

    C[6] = make_arr2(k-kcap,mcap);
    C[7] = make_arr2(ncap,mcap);
    copy_arr2(C[7],D[0]);
    C[8] = make_arr2(k-kcap,ncap);
    copy_arr2(C[8],D[5]);
    sueqar(C[8],D[7]);

    C[9] = make_arr2(kcap,m-mcap);
    C[10] = make_arr2(n-ncap,m-mcap);
    copy_arr2(C[10],D[3]);
    C[11] = make_arr2(kcap,n-ncap);
    copy_arr2(C[11],D[4]);
    negate(C[11]);
    adeqar(C[11],D[6]);

    C[12] = make_arr2(k-kcap, mcap);
    C[13] = make_arr2(n-ncap,mcap);
    copy_arr2(C[13],D[0]);
    adeqar(C[13],D[1]);
    C[14] = make_arr2(k-kcap,n-ncap);
    copy_arr2(C[14],D[7]);

    C[15] = make_arr2(kcap,mcap);
    C[16] = make_arr2(ncap,mcap);
    copy_arr2(C[16],D[0]);
    negate(C[16]);
    adeqar(C[16],D[2]);
    C[17] = make_arr2(kcap,ncap);
    copy_arr2(C[17],D[4]);
    adeqar(C[17],D[5]);

    C[18] = make_arr2(kcap,mcap);
    C[19] = make_arr2(n-ncap,mcap);
    copy_arr2(C[19],D[1]);
    sueqar(C[19],D[3]);
    C[20] = make_arr2(kcap,n-ncap);
    copy_arr2(C[20],D[6]);
    adeqar(C[20],D[7]);

    for(int i=0; i<8; i++) {
      free_arr2(D[i]);
    }

    //spawn threads/do computations

    if(spawn) {
      thread_data_t* data[THREADS];
      for(int i=0; i<THREADS-1; i++) {
        data[i] = (thread_data_t*)malloc(sizeof(thread_data_t));
        data[i]->arrs = C;
        data[i]->thread_number = i;
        pthread_create(&t[i],NULL,thread_init,data[i]);
      }
      data[THREADS-1] = (thread_data_t*)malloc(sizeof(thread_data_t));
      data[THREADS-1]->arrs = C;
      data[THREADS-1]->thread_number = THREADS-1;
      thread_init(data[THREADS-1]);

      for(int i=0; i<THREADS-1; i++)
        pthread_join(t[i],NULL);

      // printf("\n");
    } else {
      for(int i=0; i<21; i+=3)
        strassen(C+i);
    }

    //fill result with computded data

    D[0] = make_arr2(kcap,mcap);

    copy_arr2(D[0],C[0]);
    adeqar(D[0],C[9]);
    adeqar(D[0],C[18]);
    sueqar(D[0],C[12]);
    fill(result,D[0],0,0,kcap,mcap);

    resize(D[0],k-kcap,mcap);

    copy_arr2(D[0],C[12]);
    adeqar(D[0],C[6]);
    fill(result,D[0],kcap,0,k,mcap);

    resize(D[0],kcap,m-mcap);

    copy_arr2(D[0],C[3]);
    adeqar(D[0],C[9]);
    fill(result,D[0],0,mcap,kcap,m);

    resize(D[0],kcap,mcap);

    copy_arr2(D[0],C[0]);
    sueqar(D[0],C[3]);
    adeqar(D[0],C[6]);
    adeqar(D[0],C[15]);
    fill(result,D[0],kcap,mcap,k,m);
    free_arr2(D[0]);

    for(int i=0; i<21; i++)
      free_arr2(C[i]);

  } else {
    int size_test;
    size_test = ((m>cap) << 2) | ((n>cap) << 1) | (k>cap);

    //switch starts here
    //vvv
    //vvv

    //This section HAS NOT been bug checked. However it is rather unlikely to be used

    switch (size_test) {
      case (1):
        printf("1\n");
        C[0] = make_arr2(cap,m);
        C[1] = make_arr2(n,m);
        fill(C[1],A,0,0,n,m);               //[A1][B1, B2]= [R1, R2]
        C[2] = make_arr2(cap,n);            //R1 = A1B1
        fill(C[2],B,0,0,cap,n);             //R2 = A1B2

        printf("-\n");

        C[3] = make_arr2(k-cap,m);

        printf("-\n");
        C[4] = C[1];

        printf("-\n");
        C[5] = make_arr2(k-cap,n);

        printf("-\n");
        fill(C[5],B,cap,0,k,n);

        printf("-\n");

        strassen(C);
        strassen(C+3);
        fill(result,C[0],0,0,cap,m);  //R1
        fill(result,C[3],cap,0,k,m);  //R2

        printf("-\n");

        for(int i=0; i<6; i++)
          free_arr2(C[i]);

        break;
      case (2):
        printf("2\n");
        C[0] = make_arr2(k,m);
        C[1] = make_arr2(cap,m);
        fill(C[1],A,0,0,cap,m);             //[A1, A2][B1]= [R1 +R2]
        C[2] = make_arr2(k,cap);            //        [B2]
        fill(C[2],B,0,0,k,cap);             //R1 = A1B1
                                            //R2 = A2B2
        C[3] = make_arr2(k,m);
        C[4] = make_arr2(n-cap,m);
        fill(C[4],A,cap,0,n,m);
        C[5] = make_arr2(k,n-cap);
        fill(C[5],B,0,cap,k,n);

        strassen(C);
        strassen(C+3);
        fill(result,C[0],0,0,k,m);  //R1
        adeqar(result,C[3]);        //+R2

        for(int i=0; i<6; i++)
          free_arr2(C[i]);
        
        break;
      case (3):
        printf("3\n");
        C[0] = make_arr2(cap,m);
        C[1] = make_arr2(cap,m);
        fill(C[1],A,0,0,cap,m);             //[A1, A2][B1, B2]= [R1 +R2, R3 +R4]
        C[2] = make_arr2(cap,cap);          //        [B3, B4]
        fill(C[2],B,0,0,cap,cap);           //R1 = A1B1
                                            //R2 = A2B3
        C[3] = make_arr2(cap,m);            //R3 = A1B2
        C[4] = make_arr2(n-cap,m);          //R4 = A2B4
        fill(C[4],A,cap,0,n,m);
        C[5] = make_arr2(cap,n-cap);
        fill(C[5],B,0,cap,cap,n);

        C[6] = make_arr2(k-cap,m);
        C[7] = C[1];
        C[8] = make_arr2(k-cap,cap);
        fill(C[8],B,cap,0,k,cap);

        C[9] = make_arr2(k-cap,m);
        C[10] = C[4];
        C[11] = make_arr2(k-cap,n-cap);
        fill(C[11],B,cap,cap,k,n);

        strassen(C);
        strassen(C+3);
        strassen(C+6);
        strassen(C+9);
        adeqar(C[0],C[3]);            //+R2
        adeqar(C[6],C[9]);            //+R4
        fill(result,C[0],0,0,cap,m);  //R1
        fill(result,C[6],cap,0,k,m);  //R3

        for(int i=0; i<12; i++)
          free_arr2(C[i]);

        break;
      case (4):
        printf("4\n");
        C[0] = make_arr2(k,cap);
        C[1] = make_arr2(n,cap);
        fill(C[1],A,0,0,n,cap);             //[A1][B1]= [R1]
        C[2] = make_arr2(k,n);              //[A2]      [R2]
        fill(C[2],B,0,0,k,n);               //R1 = A1B1
                                            //R2 = A2B1
        C[3] = make_arr2(k,m-cap);
        C[4] = make_arr2(n,m-cap);
        fill(C[4],A,0,cap,n,m);
        C[5] = C[2];

        strassen(C);
        strassen(C+3);
        fill(result,C[0],0,0,k,cap);  //R1
        fill(result,C[3],0,cap,k,m);  //R2

        for(int i=0; i<6; i++)
          free_arr2(C[i]);

        break;
      case (5):
        printf("5\n");
        C[0] = make_arr2(cap,cap);
        C[1] = make_arr2(n,cap);
        fill(C[1],A,0,0,n,cap);             //[A1][B1, B2]= [R1, R2]
        C[2] = make_arr2(cap,n);            //[A2]          [R3, R4]
        fill(C[2],B,0,0,cap,n);             //R1 = A1B1
                                            //R2 = A1B2
        C[3] = make_arr2(k-cap,cap);        //R3 = A2B1
        C[4] = C[1];                        //R4 = A2B2
        C[5] = make_arr2(k-cap,n);
        fill(C[5],B,cap,0,k,n);

        C[6] = make_arr2(cap,m-cap);
        C[7] = make_arr2(n,m-cap);
        fill(C[7],A,0,cap,n,m);
        C[8] = C[2];

        C[9] = make_arr2(k-cap,m-cap);
        C[10] = C[7];
        C[11] = C[5];

        strassen(C);
        strassen(C+3);
        strassen(C+6);
        strassen(C+9);
        fill(result,C[0],0,0,cap,cap);  //R1
        fill(result,C[3],cap,0,k,cap);  //R2
        fill(result,C[6],0,cap,cap,m);  //R3
        fill(result,C[9],cap,cap,k,m);  //R4

        for(int i=0; i<12; i++)
          free_arr2(C[i]);

        break;
      case (6):
        printf("6\n");
        C[0] = make_arr2(k,cap);
        C[1] = make_arr2(cap,cap);
        fill(C[1],A,0,0,cap,cap);           //[A1, A2][B1]= [R1 +R2]
        C[2] = make_arr2(k,cap);            //[A3, A4][B2]  [R3 +R4]
        fill(C[2],B,0,0,k,cap);             //R1 = A1B1
                                            //R2 = A2B2
        C[3] = make_arr2(k,cap);            //R3 = A3B1
        C[4] = make_arr2(n-cap,cap);        //R4 = A4B2
        fill(C[4],A,cap,0,n,cap);
        C[5] = make_arr2(k,n-cap);
        fill(C[5],B,0,cap,k,n);

        C[6] = make_arr2(k,m-cap);
        C[7] = make_arr2(cap,m-cap);
        fill(C[7],A,0,cap,cap,n);
        C[8] = C[2];

        C[9] = make_arr2(k,m-cap);
        C[10] = make_arr2(n-cap,m-cap);
        fill(C[10],A,cap,cap,n,m);
        C[11] = C[5];

        strassen(C);
        strassen(C+3);
        strassen(C+6);
        strassen(C+9);
        adeqar(C[0],C[3]);            //+R2
        adeqar(C[6],C[9]);            //+R4
        fill(result,C[0],0,0,k,cap);  //R1
        fill(result,C[6],0,cap,k,m);  //R3

        for(int i=0; i<12; i++)
          free_arr2(C[i]);

        break;
      default: 
        fprintf(stderr,"error\n");
    }
  }

  return NULL;
}

#endif
