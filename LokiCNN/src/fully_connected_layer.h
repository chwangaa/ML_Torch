/*! \file fully_connected_layer.h
    \brief FC layer
*/

#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"
#include "math_functions.h"

typedef Layer fc_layer_t;


/*! 
    \brief FC layer forward routine
    \param l the FC layer pointer
    \param in the input images
    \param out the ouput images
    \param start the index in the input batch from which to perform forwarding
    \param end the index in the input batch from which to stop perform forwarding

    It is not difficult to see that FC layer is by nature matrix-vector multiplication, hence calling gemv is default
*/
void fc_forward(const fc_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end) {
  for (int j = start; j <= end; j++) {
    const vol_t* V = in[j];
    vol_t* A = out[j];

    int num_inputs = l->num_inputs;
    storage_t* outputs = out[j]->w;
    storage_t* biases = l->biases->w;
    // assign bias here
    for(int i = 0; i < l->out_depth; i++){
      outputs[i] = biases[i];
    }
    
    /* this is effectively an matrix-vector multiplication */
    cblas_gemv(l->out_depth, num_inputs,
               l->filters_flat->w, num_inputs,
               V->w,
               out[j]->w);
  }
}

/*! 
    \brief constructor of fully connected layer
    \param in_sx the width of input image
    \param in_sy the height of input image
    \param in_depth the channel number of input image
    \param out_depth the number of outputs
*/
fc_layer_t* make_fc_layer(int in_sx, int in_sy, int in_depth,
                          int out_depth) {
  fc_layer_t* l = (fc_layer_t*)malloc(sizeof(fc_layer_t));

  // required
  l->type = FULLY_CONNECTED;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
  l->out_depth = out_depth;


  // computed
  l->num_inputs = l->in_sx * l->in_sy * l->in_depth;
  l->out_sx = 1;
  l->out_sy = 1;

  l->filters_flat = make_vol(in_sx*in_sy*in_depth, out_depth, 1, 0);
  l->biases = make_vol(1, 1, l->out_depth, 0);

  l->forward = &fc_forward;
   
  return l;
}

/*! 
    \brief default fully connected layer weight loader
    \param l pointer to the FC layer
    \param params specifiers of FC layer parameters, used to track whether the weight file matches the layer's requirement
    \param weights pointer to the starting point of weight array

    this is just one way of loading values, one can easily define its own loader according to specific needs
    for loading from weights trained by caffe, this is an easy approach
*/
void fc_load(fc_layer_t* l, const int* params, const weight_t* weights) {
  
  int num_inputs = params[0];
  int out_depth = params[1];
  assert(out_depth == l->out_depth);
  assert(num_inputs == l->num_inputs);

  int weight = 0;
  int index = 0;
  for(int i = 0; i < l->out_depth; i++)
    for(int d = 0; d < l->num_inputs; d++) {
      double val = weights[weight++];
      l->filters_flat->w[index++] = readDouble(val);
    }

  for(int i = 0; i < l->out_depth; i++) {
    double val = weights[weight++];
    l->biases->w[i] = readDouble(val);
  }

}

#endif
