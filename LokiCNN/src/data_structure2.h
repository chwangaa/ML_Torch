#ifndef DATA2_H
#define DATA2_H

/*
  This extra data_structure is used for Strassen's Algorithm
*/

#include "data_structure.h"

typedef struct arr2 {
  int sx,sy;
  storage_t* data;
} arr2_t;

static inline storage_t get_arr2(arr2_t *arr, int x, int y) {
  return arr->data[y * (arr->sx) + x];
}

static inline void set_arr2(arr2_t *arr, int x, int y, storage_t v) {
  arr->data[y * (arr->sx) + x] = v;
}

static arr2_t* make_arr2(int sx, int sy) {
  arr2_t* result = (arr2_t*)malloc(sizeof(arr2_t));
  result->data = (storage_t*)malloc(sizeof(storage_t)*sx*sy);
  result->sx = sx;
  result->sy = sy;

  return result;
}

void free_arr2(arr2_t* arr) {
  free(arr->data);
  free(arr);
}

static void fill(arr2_t* arr, arr2_t* data, int startx, int starty, int endx, int endy) {
  int index = 0;
  if(arr->sx>data->sx || arr->sy>data->sy) {
    for(int i=starty, ay=0; i<endy; i++, ay++) {
      for(int j=startx, ax=0; j<endx; j++, ax++) {
        set_arr2(arr,j,i,get_arr2(data,ax,ay)); //small optimisation possible
      }
    }
  } else {
    for(int i=starty, ay=0; i<endy; i++, ay++) {
      for(int j=startx, ax=0; j<endx; j++, ax++) {
        set_arr2(arr,ax,ay,get_arr2(data,j,i)); //small optimisation possible
      }
    }
  }
}

static void copy_arr2(arr2_t* dest, arr2_t* src) {
  if(dest->sy == src->sy && dest->sx == src->sx) {
    int index = 0;
    for(int y=0; y<dest->sy; y++)
      for(int x=0; x<dest->sx; x++, index++)
        dest->data[index] = src->data[index];
  } else {
    fill(dest,src,0,0,dest->sx,dest->sy);
  }
}

static void print_arr2(arr2_t* arr) {
  for(int y=0; y<arr->sy; y++) {
    for(int x=0; x<arr->sx; x++) {
      printf("%f ",get_arr2(arr,x,y));
    }
    printf("\n");
  }
}

#endif //ndef data2_h
