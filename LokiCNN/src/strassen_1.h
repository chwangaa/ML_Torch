// Strassen implementation ----------------------------------------------------
#ifndef STRASSEN_H
#define STRASSEN_H

#define CACHE_SIZE 128*1024

#include <assert.h>
#include <math.h>

#include "data_structure_2.h"

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

arr2_t* strassen(arr2_t *A, arr2_t *B) {
  //printf("%d,%d,%d,%d\n",A->sx,A->sy,B->sx,B->sy);
  int m = A->sy;
  int n = A->sx;
  assert(n == B->sy);
  int k = B->sx;
  arr2_t* result = make_arr2(k,m);

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
    return result;
  }

  if(size == 1) {
    set_arr2(result,0,0, get_arr2(A,0,0) * get_arr2(B,0,0));
    return result;
  }
 
  int cap = (size+1)>>1;

  arr2_t *M[7];     //store results of multiplications
  arr2_t *D[8];     //store re-used data
  arr2_t *C[2];     //re-used memory for once-used data

  C[0] = make_arr2(cap,cap);
  C[1] = make_arr2(cap,cap);

  if(m>1 && n>1 && k>1) {
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

    resize(C[0],ncap,mcap);
    copy_arr2(C[0],D[0]);
    adeqar(C[0],D[3]);
    resize(C[1],kcap,ncap);
    copy_arr2(C[1],D[4]);
    adeqar(C[1],D[7]);
    M[0] = strassen(C[0],C[1]);

    resize(C[0],ncap,m-mcap);
    copy_arr2(C[0],D[2]);
    adeqar(C[0],D[3]);
    M[1] = strassen(C[0],D[4]);

    resize(C[1],k-kcap,ncap);
    copy_arr2(C[1],D[5]);
    sueqar(C[1],D[7]);
    M[2] = strassen(D[0],C[1]);

    resize(C[1],kcap,n-ncap);
    copy_arr2(C[1],D[4]);
    negate(C[1]);
    adeqar(C[1],D[6]);
    M[3] = strassen(D[3],C[1]);

    resize(C[0],n-ncap,mcap);
    copy_arr2(C[0],D[0]);
    adeqar(C[0],D[1]);
    M[4] = strassen(C[0],D[7]);

    resize(C[0],ncap,mcap);
    copy_arr2(C[0],D[0]);
    negate(C[0]);
    adeqar(C[0],D[2]);
    resize(C[1],kcap,ncap);
    copy_arr2(C[1],D[4]);
    adeqar(C[1],D[5]);
    M[5] = strassen(C[0],C[1]);

    resize(C[0],n-ncap,mcap);
    copy_arr2(C[0],D[1]);
    sueqar(C[0],D[3]);
    resize(C[1],kcap,n-ncap);
    copy_arr2(C[1],D[6]);
    adeqar(C[1],D[7]);
    M[6] = strassen(C[0],C[1]);

    for(int i=0; i<8; i++) {
      free_arr2(D[i]);
    }

    resize(C[0],kcap,mcap);

    copy_arr2(C[0],M[0]);
    adeqar(C[0],M[3]);
    adeqar(C[0],M[6]);
    sueqar(C[0],M[4]);
    fill(result,C[0],0,0,kcap,mcap);
    free_arr2(M[6]);

    resize(C[0],k-kcap,mcap);

    copy_arr2(C[0],M[4]);
    adeqar(C[0],M[2]);
    fill(result,C[0],kcap,0,k,mcap);
    free_arr2(M[4]);

    resize(C[0],kcap,m-mcap);

    copy_arr2(C[0],M[1]);
    adeqar(C[0],M[3]);
    fill(result,C[0],0,mcap,kcap,m);
    free_arr2(M[3]);

    resize(C[0],kcap,mcap);

    copy_arr2(C[0],M[0]);
    sueqar(C[0],M[1]);
    adeqar(C[0],M[2]);
    adeqar(C[0],M[5]);
    fill(result,C[0],kcap,mcap,k,m);
    free_arr2(M[0]);
    free_arr2(M[1]);
    free_arr2(M[2]);
    free_arr2(M[5]);
  } else {
    int size_test;
    size_test = ((m>cap) << 2) | ((n>cap) << 1) | (k>cap);

    //switch starts here
    //vvv
    //vvv

    switch (size_test) {
      case (1):
        resize(C[0],n,m);
        fill(C[0],A,0,0,n,m);               //[A1][B1, B2]= [R1, R2]
        resize(C[1],cap,n);                 //R1 = A1B1
        fill(C[1],B,0,0,cap,n);             //R2 = A1B2

        M[0] = strassen(C[0],C[1]);
        fill(result,M[0],0,0,cap,m);  //R1
        free_arr2(M[0]);

        resize(C[1],k-cap,n);
        fill(C[1],B,cap,0,k,n);

        M[0] = strassen(C[0],C[1]);
        fill(result,M[0],cap,0,k,m);  //R2
        free_arr2(M[0]);
        break;
      case (2):
        resize(C[0],cap,m);
        fill(C[0],A,0,0,cap,m);             //[A1, A2][B1]= [R1 +R2]
        resize(C[1],k,cap);                 //        [B2]
        fill(C[1],B,0,0,k,cap);             //R1 = A1B1
                                            //R2 = A2B2
        M[0] = strassen(C[0],C[1]);
        fill(result,M[0],0,0,k,m);  //R1
        free_arr2(M[0]);

        resize(C[0],n-cap,m);
        fill(C[0],A,cap,0,n,m);
        resize(C[1],k,n-cap);
        fill(C[1],B,0,cap,k,n);

        M[0] = strassen(C[0],C[1]);
        adeqar(result,M[0]);        //+R2
        free_arr2(M[0]);
        break;
      case (3):
        D[0] = make_arr2(cap,m);
        fill(D[0],A,0,0,cap,m);             //[A1, A2][B1, B2]= [R1 +R2, R3 +R4]
        D[1] = make_arr2(n-cap,m);          //        [B3, B4]
        fill(D[1],A,cap,0,n,m);             //R1 = A1B1
                                            //R2 = A2B3
        fill(C[0],B,0,0,cap,cap);           //R3 = A1B2
        resize(C[1],cap,n-cap);             //R4 = A2B4
        fill(C[1],B,0,cap,cap,n);
        M[0] = strassen(D[0],C[0]);
        M[1] = strassen(D[1],C[1]);
        adeqar(M[0],M[1]);            //+R2
        fill(result,M[0],0,0,cap,m);  //R1
        free_arr2(M[0]);
        free_arr2(M[1]);

        resize(C[0],k-cap,cap);
        fill(C[0],B,cap,0,k,cap);
        resize(C[1],k-cap,n-cap);
        fill(C[1],B,cap,cap,k,n);
        M[0] = strassen(D[0],C[0]);
        free_arr2(D[0]);
        M[1] = strassen(D[1],C[1]);
        free_arr2(D[1]);
        adeqar(M[0],M[1]);            //+R4
        fill(result,M[0],cap,0,k,m);  //R3
        free_arr2(M[0]);
        free_arr2(M[1]);
        break;
      case (4):
        resize(C[0],n,cap);
        fill(C[0],A,0,0,n,cap);             //[A1][B1]= [R1]
        resize(C[1],k,n);                   //[A2]      [R2]
        fill(C[1],B,0,0,k,n);               //R1 = A1B1
        M[0] = strassen(C[0],C[1]);         //R2 = A2B1
        fill(result,M[0],0,0,k,cap);  //R1
        free_arr2(M[0]);

        resize(C[0],n,m-cap);
        fill(C[0],A,0,cap,n,m);
        M[0] = strassen(C[0],C[1]);
        fill(result,M[0],0,cap,k,m);  //R2
        free_arr2(M[0]);
        break;
      case (5):
        D[0] = make_arr2(n,cap);            //[A1][B1, B2]= [R1, R2]
        fill(D[0],A,0,0,n,cap);             //[A2]          [R3, R4]
        D[2] = make_arr2(n,m-cap);          //R1 = A1B1
        fill(D[2],A,0,cap,n,m);             //R2 = A1B2
        resize(C[0],cap,n);                 //R3 = A2B1
        fill(C[0],B,0,0,cap,n);             //R4 = A2B2
        M[0] = strassen(D[0],C[0]);
        fill(result,M[0],0,0,cap,cap);  //R1
        free_arr2(M[0]);

        M[0] = strassen(D[2],C[0]);
        fill(result,M[0],0,cap,cap,m);  //R3
        free_arr2(M[0]);

        resize(C[0],k-cap,n);
        fill(C[0],B,cap,0,k,n);
        M[0] = strassen(D[0],C[0]);
        fill(result,M[0],cap,0,k,cap);  //R2
        free_arr2(M[0]);
        free_arr2(D[0]);

        M[0] = strassen(D[2],C[0]);
        fill(result,M[0],cap,cap,k,m);  //R4
        free_arr2(M[0]);
        free_arr2(D[2]);
        break;
      case (6):
        D[4] = make_arr2(k,cap);            //[A1, A2][B1]= [R1 +R2]
        fill(D[4],B,0,0,k,cap);             //[A3, A4][B2]  [R3 +R4]
        D[6] = make_arr2(k,n-cap);          //R1 = A1B1
        fill(D[6],B,0,cap,k,n);             //R2 = A2B2
        fill(C[0],A,0,0,cap,cap);           //R3 = A3B1
        resize(C[1],n-cap,cap);             //R4 = A4B2
        fill(C[1],A,cap,0,n,cap);
        M[0] = strassen(C[0],D[4]);
        M[1] = strassen(C[1],D[6]);
        adeqar(M[0],M[1]);            //+R2
        fill(result,M[0],0,0,k,cap);  //R1
        free_arr2(M[0]);
        free_arr2(M[1]);

        resize(C[0],cap,m-cap);
        fill(C[0],A,0,cap,cap,m);
        resize(C[1],n-cap,m-cap);
        fill(C[1],A,cap,cap,n,m);
        M[0] = strassen(C[0],D[4]);
        free_arr2(D[4]);
        M[1] = strassen(C[1],D[6]);
        free_arr2(D[6]);
        adeqar(M[0],M[1]);            //+R4
        fill(result,M[0],0,cap,k,m);  //R3
        free_arr2(M[0]);
        free_arr2(M[1]);
        break;
      default: 
        fprintf(stderr,"error\n");
    }

    free_arr2(C[0]);
    free_arr2(C[1]);
  }

  return result;
}

#endif
