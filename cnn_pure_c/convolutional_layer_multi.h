// Convolutional Layer --------------------------------------------------------
#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "layer.h"
#include "strassen_mult.h"

typedef Layer conv_layer_t;

void conv_forward_with_padding(conv_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
        
    const int V_sx = V->sx;
    const int V_sy = V->sy;
    const int xy_stride = l->stride;

    const int outx = l->out_sx;
    const int outy = l->out_sy;
  } 
}


void conv_forward_without_padding(conv_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];

    const int V_sx = V->sx;
    const int V_sy = V->sy;
    const int V_d = V->depth;
    const int xy_stride = l->stride;
    vol_t **filters = l->filters;

    const int kx = l->sx;
    const int ky = l->sy;

    const int A_d = A->depth;
    const int outx = l->out_sx;
    const int outy = l->out_sy;

    int weightx = l->sx * l->sy * V->depth;
    int weighty = A->depth;                
    int datax = outx * outy;
    int datay = weightx;

    arr2_t* arr[3];
    arr[0] = make_arr2(datax, weighty);
    arr[1] = make_arr2(weightx, weighty);
    arr[2] = make_arr2(datax, datay);

    for(int out_d=0; out_d<A_d; out_d++)
      for(int in_d=0; in_d<V_d; in_d++)
        for(int y=0; y<ky; y++)
          for(int x=0; x<kx; x++) {
            set_arr2(arr[1],(in_d*ky + y)*kx + x,out_d,
                        get_vol(filters[out_d],x,y,in_d));
          }

    for(int ay=0; ay<outy; ay++)
      for(int ax=0; ax<outx; ax++)
        for(int in_d=0; in_d<V_d; in_d++)
          for(int y=0; y<ky; y++)
            for(int x=0; x<kx; x++) {
              set_arr2(arr[2],ay*outx + ax,(in_d*ky + y)*kx + x,
                      get_vol(V,x+ax*xy_stride,y+ay*xy_stride,in_d));
            }

    top = 1;

    strassen(arr);

    for(int out_d=0; out_d<A_d; out_d++)
      for(int ay=0; ay<outy; ay++)
        for(int ax=0; ax<outx; ax++)
          set_vol(A,ax,ay,out_d,get_arr2(arr[0],ay*outx+ax,out_d));

    for(int i=0; i<3; i++)
      free_arr2(arr[i]);
  }
}


conv_layer_t* make_conv_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int filters, int stride, int pad) {
  conv_layer_t* l = (conv_layer_t*)malloc(sizeof(conv_layer_t));

  // required
  l->sx = sx;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
  l->out_depth = filters;
    
  // optional
  l->sy = l->sx;
  l->stride = stride;
  l->pad = pad;
  l->l1_decay_mul = 0.0;
  l->l2_decay_mul = 1.0;

  //
  assert(l->pad >=0);

  // computed
  l->out_sx = floor((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = floor((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*filters);
  for (int i = 0; i < filters; i++) {
    l->filters[i] = make_vol(l->sx, l->sy, l->in_depth, 0.0);
  }

  l->bias = 0.0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  if(l->pad != 0){
    l->forward = &conv_forward_with_padding;
  }
  else{
    l->forward = &conv_forward_without_padding;   
  }
  return l;
}

void conv_load_file(conv_layer_t* l, const char* fn) {
  FILE* fin = fopen(fn, "r");
  assert(fin != NULL);
  
  int items_read;
  int sx, sy, depth, filters;
  sx = sy = depth = filters = 0;
  items_read = fscanf(fin, "%d %d %d %d", &sx, &sy, &depth, &filters);
  assert(items_read == 4);  
  assert(sx == l->sx);
  assert(sy == l->sy);
  assert(depth == l->in_depth);
  assert(filters == l->out_depth);

  for(int d = 0; d < l->out_depth; d++)
    for (int z = 0; z < depth; z++)
      for (int x = 0; x < sx; x++)
        for (int y = 0; y < sy; y++){
          double val;
          items_read = fscanf(fin, "%lf", &val);
          assert(items_read == 1);
          // fprintf(stderr, "value read is %f \n", val);
          set_vol(l->filters[d], x, y, z, val);
        }

  // fprintf(stderr, "weights loaded correctly \n");

  for(int d = 0; d < l->out_depth; d++) {
    double val;
    items_read = fscanf(fin, "%lf", &val);
    assert(items_read == 1);
    set_vol(l->biases, 0, 0, d, val);
  }

  fclose(fin);
}

void conv_load(conv_layer_t* l, const int* params, const weight_t* weights) {  
  int sx, sy, depth, filters;
  sx = params[0]; sy = params[1]; depth = params[2]; filters = params[3];
  assert(sx == l->sx);
  assert(sy == l->sy);
  assert(depth == l->in_depth);
  assert(filters == l->out_depth);

  int i=0;
  for(int d = 0; d < l->out_depth; d++)
    for (int z = 0; z < depth; z++)
      for (int x = 0; x < sx; x++)
        for (int y = 0; y < sy; y++){
          weight_t val = weights[i++];
          // fprintf(stderr, "value read is %f \n", val);
          set_vol(l->filters[d], x, y, z, val);
        }

  // fprintf(stderr, "weights loaded correctly \n");

  for(int d = 0; d < l->out_depth; d++) {
    weight_t val = weights[i++];
    set_vol(l->biases, 0, 0, d, val);
  }

}

#endif
