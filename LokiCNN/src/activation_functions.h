/*! \file activation_functions.h
    \brief a list of available activation functions
*/

#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "layer.h"

/// ReLU functions iterates over the entire image, and does a max(p, 0) operation on every pixels
void relu_func(Layer* l, vol_t** input, const int start, const int end){
  for (int j = start; j <= end; j++) {
    storage_t* V = input[j]->w;
    for (int i = 0; i < l->out_sx*l->out_sy*l->out_depth; i++) {
      if(V[i] < 0){
      	V[i] = 0;
      }
    }
  }
}

// #ifdef LOKI

// void relu_func(Layer* l, vol_t** input, const int start, const int end){
//   for (int j = start; j <= end; j++) {
//     storage_t* V = input[j]->w;
//     int size = l->out_sx * l->out_sy * l->out_depth;
//     int i = 0;
//     for (; i < size - 4 ; i+= 4) {
//       // V->w[i] = (V->w[i] < 0) ? 0 : V->w[i];
//       // V->w[i+1] = (V->w[i+1] < 0) ? 0 : V->w[i+1];
//       // V->w[i+2] = (V->w[i+2] < 0) ? 0 : V->w[i+2];
//       // V->w[i+3] = (V->w[i+3] < 0) ? 0 : V->w[i+3];
//       asm volatile(
//           "fetchr.eop 0f \n"
//           "0:"
//           "fetchr 1f \n"
//           "addui %0, %0 -4 \n"
//           "addui %0, %0, %2 -> 10 \n"
//           "addui %0, %0, %2 -> 11 \n"
//           "addui %0, %0, %2 -> 12 \n"
//           "addui %0, %0, %2 -> 13 \n"

//           "stw r4, 0x0(%1) -> 10\n"
//           "stw r5, 0x4(%1) -> 11\n"
//           "stw r6, 0x8(%1) -> 12\n"
//           "stw.eop r7, 0xc(%1) ->13\n"
//           "1:"
//           : 
//           : "r"(&V[i])
//         );
//     }
//     for(; i < size; i++){
//       V[i] = (V[i] < 0) ? 0 : V[i];
//     }
//   }
// }

// #endif
//TODO: other activation functions like Tanh, shouldn't be difficult to come up

#endif