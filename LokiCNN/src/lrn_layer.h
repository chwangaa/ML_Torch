//Normalisation Layer ---------------------------------------------------------
#ifndef LRN_LAYER_H
#define LRN_LAYER_H

#include "layer.h"

typedef Layer lrn_layer_t;

void lrn_accross_channel_forward(lrn_layer_t* l, vol_t** in, 
                          vol_t** out, int start, int end) {
  for(int i=start; i<=end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];

    for(int d=0; d<l->out_depth; d++) {
        int start = d-(l->sx)/2;
        start = fmax(0,start);
        int end = d+(l->sx)/2;
        end = fmin(l->out_depth-1,end);
      for(int ax=0; ax<l->out_sx; ax++) {
        for(int ay=0; ay<l->out_sy; ay++) {
          double accum = 0;

          for(int od=start; od<=end; od++) {
            double val = get_vol(V, ax, ay, od);
            accum += val*val;
          }

          accum *= l->alpha / l->sx;
          accum++;
          accum = pow(accum, l->beta);
          set_vol(A, ax, ay, d, get_vol(V,ax,ay,d)/accum);
        }
      }
    }
  }
}

void lrn_within_channel_forward(lrn_layer_t* l, vol_t** in, 
                vol_t** out, int start, int end) {
  const int KERNEL_SIZE = l->sx * l->sx;

  for(int i=start; i<=end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];

    for(int d=0;d<l->out_depth;d++) {
      int x = -l->pad;
      for(int ax=0; ax<l->out_sx; x++,ax++) {
        int y = -l->pad;
        for(int ay=0; ay<l->out_sy; y++,ay++) {
  
          storage_t accum = 0;

          for(int fy=0; fy<l->sx; fy++) {
            for(int fx=0; fx<l->sx; fx++) {
              int oy = y+fy;
              int ox = x+fx;
              if(oy>=0 && oy<V->sy && ox>=0 && ox<V->sx) {
                storage_t val = get_vol(V,ox,oy,d);
                accum += val*val;
              }
            }
          }
          accum *= l->alpha / KERNEL_SIZE;
          accum++;
          accum = pow(accum, l->beta);
          set_vol(A, ax, ay, d, get_vol(V,ax,ay,d)/accum);
        }
      }
    }
  }
}

lrn_layer_t* make_lrn_layer(int in_sx, int in_sy, int in_depth,
    int local_size, double alpha, double beta,
    char across_channel_flag) {
  lrn_layer_t* l = (lrn_layer_t*)malloc(sizeof(lrn_layer_t));

  // required
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
  l->out_depth = in_depth;
  l->out_sx = in_sx;
  l->out_sy = in_sy;
  l->sx = local_size;
  l->alpha = alpha;
  l->beta = beta;

  l->forward = across_channel_flag ? &lrn_accross_channel_forward :
                                     &lrn_within_channel_forward;

  //optional
  l->type = LRN;
  l->pad = l->sx/2;

  return l;
}

#endif
