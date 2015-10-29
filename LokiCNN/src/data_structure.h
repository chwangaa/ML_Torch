/*! \file data_structure.h
    \brief definition of images
*/

#ifndef DATA_H
#define DATA_H
#include "setting.h"
#include "math_functions.h"
/*
 * Represents a three-dimensional array of numbers, and its size. The numbers
 * at (x,y,d) are stored in array w at location ((v->sx * y)+x)*v->depth+d.
 */
typedef struct vol {
  uint64_t sx,sy,depth;
  storage_t* w;
} vol_t;

/*!
    \brief return the value of pixel from specified location
    \param v the image pointer
    \param x specifies the width
    \param y specifies the height
    \param d specifies the channel number, or the depth
    
    note that memory is row-major, as it is in Caffe's implementation. So the last dimension changes fastest,
    
    index (x, y, d) physically correspond to (X*d + x)*Y + y
*/
inline storage_t get_vol(const vol_t* v, int x, int y, int d) {
  return v->w[((v->sx * d) +x)*(v->sy) + y];
}

static inline int get_vol_index(const vol_t* v, int x, int y, int d){
  return ((v->sy * x) + y) * (v->depth) + d;
}

/**
  * \brief print the image pixels at specified depth
  * 
  * the printed value will be in double
  * if FIXED_POINT is used, values are converted to double for direct comparison
  */
void print_vol(const vol_t* v, int depth){
  int sx = v->sx;
  int sy = v->sy;
  for(int x =0; x < sx; x++){
    for(int y =0; y < sy; y++){
      storage_t val = get_vol(v, x, y, depth);
      #if defined(FIX16)
         fprintf(stderr, "%lf", fix16_to_dbl(val));
      #elif defined(FIX8)
         fprintf(stderr, "%lf", fix8_to_dbl(val));
      #else
         fprintf(stderr, "%lf", val);
      #endif
    }
    fprintf(stderr, "\n\n");
  }
}

/*
 * Get the value at a specific entry of the array.
 */
static inline void set_vol(vol_t* v, int x, int y, int d, storage_t val) {
  v->w[((v->sx * d) + x) * (v->sy) + y] = val;
}

/*
 * Allocate a new array with specific dimensions and default value v.
 */
static vol_t* make_vol(int sx, int sy, int d, storage_t v) {
  vol_t* out = (vol_t*)malloc(sizeof(struct vol));
  out->w = (storage_t*)malloc(sizeof(storage_t)*(sx*sy*d));
  out->sx = sx;
  out->sy = sy;
  out->depth = d;

  if(WEIGHTS_INITIALIZE_TO_ZERO){
    for (int x = 0; x < sx; x++)
      for (int y = 0; y < sy; y++)
        for (int z = 0; z < d; z++){
          set_vol(out, x, y, z, v);
        }
  }

  return out;
}

/*
 * Copy the contents of one Volume to another (assuming same dimensions).
 */
static void copy_vol(vol_t* dest, const vol_t* src) {
  const int size = dest->sx * dest->sy * dest->depth;

  const storage_t* src_values = src->w;

  storage_t* dest_value = dest->w;

  for(int i = 0; i < size; i++){
    dest_value[i] = src_values[i];
  }
}

/**\brief destructor
 *
 */
void free_vol(vol_t* v) {
  free(v->w);
  free(v);
}

#endif
