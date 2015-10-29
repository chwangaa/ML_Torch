/*! \file soft_max_layer.h
    \brief softmax layer
*/
// Softmax Layer --------------------------------------------------------------
#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer.h"

typedef Layer softmax_layer_t;


void softmax_forward(const softmax_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end) {
  storage_t es[l->out_depth];
  for (int j = start; j <= end; j++) {
    const vol_t* V = in[j];
    vol_t* A = out[j];
    // compute max activation
    storage_t amax = V->w[0];
    for(int i=1;i<l->out_depth;i++) {
      if(V->w[i] > amax) amax = V->w[i];
    }
    
    // compute exponentials (carefully to not blow up)
    storage_t esum = 0;
    for(int i=0;i<l->out_depth;i++) {
      storage_t e = exp_t(V->w[i] - amax);
      esum = add(esum, e);
      es[i] = e;
    }
  
    // normalize and output to sum to one
    for(int i=0;i<l->out_depth;i++) {
      es[i] = divide(es[i], esum);
      A->w[i] = es[i];
    }
  }
}

softmax_layer_t* make_softmax_layer(int in_sx, int in_sy, int in_depth) {
  softmax_layer_t* l = (softmax_layer_t*)malloc(sizeof(softmax_layer_t));

  // required
  l->type = SOFTMAX;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // computed
  l->out_sx = 1;
  l->out_sy = 1;
  l->out_depth = l->in_sx * l->in_sy * l->in_depth;

  l->es = (storage_t*)malloc(sizeof(storage_t)*l->out_depth);
  l->forward = &softmax_forward;
  return l;
}

#endif