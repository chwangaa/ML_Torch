// MaxPool Layer -----------------------------------------------------------------
#ifndef AVERAGE_POOLING_LAYER_H
#define AVERAGE_POOLING_LAYER_H

#include "layer.h"

typedef Layer pool_layer_t;


void average_pool_forward(pool_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
        
    int n=0;
    const int KERNEL_SIZE = l->sx * l->sy;
    for(int d=0;d<l->out_depth;d++) {
      int x = -l->pad;
      int y = -l->pad;
      for(int ax=0; ax<l->out_sx; x+=l->stride,ax++) {
        y = -l->pad;
        for(int ay=0; ay<l->out_sy; y+=l->stride,ay++) {
  
          double accum = 0
          for(int fx=0;fx<l->sx;fx++) {
            for(int fy=0;fy<l->sy;fy++) {
              int oy = y+fy;
              int ox = x+fx;
              if(oy>=0 && oy<V->sy && ox>=0 && ox<V->sx) {
                double v = get_vol(V, ox, oy, d);
                accum += v;
              }
            }
          }
          n++;
          accum /= KERNEL_SIZE;
          set_vol(A, ax, ay, d, accum);
        }
      }
    }
  }
}

pool_layer_t* make_average_pool_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int stride) {
  pool_layer_t* l = (pool_layer_t*)malloc(sizeof(pool_layer_t));

  // required
  l->sx = sx;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // optional
  l->sy = l->sx;
  l->stride = stride;
  l->pad = 0;

  // computed
  l->out_depth = in_depth;
  l->out_sx = floor((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = floor((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  l->forward = &average_pool_forward;

  return l;
}

#endif