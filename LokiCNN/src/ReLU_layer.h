/*! \file relu_layer.h
    \brief relu layer
*/
// Relu Layer -----------------------------------------------------------------
#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

typedef Layer relu_layer_t;

/// constructor for relu layer
relu_layer_t* make_relu_layer(int in_sx, int in_sy, int in_depth) {
  relu_layer_t* l = (relu_layer_t*)malloc(sizeof(relu_layer_t));

  // required
  l->type = RELU;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;

  // computed
  l->out_sx = l->in_sx;
  l->out_sy = l->in_sy;
  l->out_depth = l->in_depth;

  /// forward function in ReLU layer is in_place, embedded within last layer
  l->forward = 0;

  return l;
}

#endif