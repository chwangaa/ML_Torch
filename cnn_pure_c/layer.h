#ifndef LAYER_H
#define LAYER_H

#include "data_structure.h"

typedef enum {
    CONVOLUTIONAL,
    POOLING,
    SOFTMAX,
    RELU
} LAYER_TYPE;


typedef struct Layer{
    LAYER_TYPE type;
    
    int sx;
    int sy;

    int in_sx;
    int in_sy;
    int in_depth;
    int num_inputs;

    int stride;
    int pad;

    int out_depth;
    int out_sx;
    int out_sy;

    double* es;

    double l1_decay_mul;
    double l2_decay_mul;

    double bias;
    vol_t* biases;
    vol_t** filters;

    void (*forward)(struct Layer* l, vol_t** in, vol_t** out, int start, int end);
} Layer;

#endif
