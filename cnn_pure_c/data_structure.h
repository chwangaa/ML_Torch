#ifndef DATA_H
#define DATA_H
/*
 * Represents a three-dimensional array of numbers, and its size. The numbers
 * at (x,y,d) are stored in array w at location ((v->sx * y)+x)*v->depth+d.
 */

typedef double storage_t;

typedef struct vol {
  uint64_t sx,sy,depth;
  storage_t* w;
} vol_t;

/*
 * Set the value at a specific entry of the array.
 */

static inline storage_t get_vol(vol_t* v, int x, int y, int d) {
  // int index = ((v->sx * y)+x)*v->depth + d;
  // int index = ((v->sx * d) +x)*(v->sy) + y;
  return v->w[((v->sx * d) +x)*(v->sy) + y];
}

/*
 * Get the value at a specific entry of the array.
 */

static inline void set_vol(vol_t* v, int x, int y, int d, storage_t val) {
  // int index = ((v->sx * y)+x)*v->depth + d;
  // int index = ((v->sx * d) +x)*(v->sy) + y;
  // fprintf(stderr, "the index is %d \n", index);
  v->w[((v->sx * d) +x)*(v->sy) + y] = val;
}

static inline void print_vol(vol_t* v, int depth){
  int sx = v->sx;
  int sy = v->sy;
  for(int x =0; x < sx; x++){
    for(int y =0; y < sy; y++){
      fprintf(stderr, "%lf  ", get_vol(v, x, y, depth));
    }
    fprintf(stderr, "\n\n");
  }
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

  for (int x = 0; x < sx; x++)
    for (int y = 0; y < sy; y++)
      for (int z = 0; z < d; z++)
        set_vol(out, x, y, z, v);

  return out;
}

/*
 * Copy the contents of one Volume to another (assuming same dimensions).
 */
static vol_t* copy_vol(vol_t* dest, vol_t* src) {
  for (int x = 0; x < dest->sx; x++)
    for (int y = 0; y < dest->sy; y++)
      for (int z = 0; z < dest->depth; z++)
        set_vol(dest, x, y, z, get_vol(src, x, y, z));
}

/*
 * Deallocate the array.
 */
void free_vol(vol_t* v) {
  free(v->w);
  free(v);
}

// Function to dump the content of a volume for comparison.
void dump_vol(vol_t* v) {
  printf("%ld,%ld,%ld", v->sx, v->sy, v->depth);
  for (int x = 0; x < v->sx; x++)
    for (int y = 0; y < v->sy; y++)
      for (int z = 0; z < v->depth; z++) {
        printf(",%.20lf", get_vol(v, x, y, z));
      }
  printf("\n");
}

#endif