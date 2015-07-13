// FC Layer -------------------------------------------------------------------
#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"

typedef Layer fc_layer_t;

void fc_forward(fc_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int j = start; j <= end; j++) {
    vol_t* V = in[j];
    vol_t* A = out[j];
        
    for(int i=0;i<l->out_depth;i++) {
      // for each output
      double a = 0.0; // initialize the accum
      for(int d=0;d<l->num_inputs;d++) {
        a += V->w[d] * l->filters[i]->w[d];
      }
      a += l->biases->w[i];
      set_vol(A, 0, 0, i, a);
    }
  }
}

fc_layer_t* make_fc_layer(int in_sx, int in_sy, int in_depth,
                          int num_neurons) {
  fc_layer_t* l = (fc_layer_t*)malloc(sizeof(fc_layer_t));

  // required
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
  l->out_depth = num_neurons;
    
  // optional
  l->l1_decay_mul = 0.0;
  l->l2_decay_mul = 1.0;

  // computed
  l->num_inputs = l->in_sx * l->in_sy * l->in_depth;
  l->out_sx = 1;
  l->out_sy = 1;

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*num_neurons);
  for (int i = 0; i < l->out_depth; i++) {
    l->filters[i] = make_vol(1, 1, l->num_inputs, 0.0);
  }

  l->bias = 0.0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  l->forward = &fc_forward;

  printf("fc: sx:N/A in_depth:%d in_sx:%d in_sy:%d out_depth:%d out_sx:%d out_sy:%d\n",
   l->in_depth, l->in_sx, l->in_sy, l->out_depth, l->out_sx, l->out_sy);
   
  return l;
}

void fc_load(fc_layer_t* l, const int* params, const weight_t* weights) {
  
  int num_inputs = params[0];
  int out_depth = params[1];
  printf("%d %d; %d %d\n", out_depth, l->out_depth, num_inputs, l->num_inputs);
  assert(out_depth == l->out_depth);
  assert(num_inputs == l->num_inputs);

  int weight;
  for(int i = 0; i < l->out_depth; i++)
    for(int d = 0; d < l->num_inputs; d++) {
      double val = weights[weight++];
      l->filters[i]->w[d] = val;
    }

  for(int i = 0; i < l->out_depth; i++) {
    double val = weights[weight++];
    l->biases->w[i] = val;
  }

}

#endif
