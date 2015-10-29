/*! \file layer.h
    \brief generic layer definition
*/

/*! \fn void (*forward)(const struct Layer* l, const vol_t** in, vol_t** out, const int start, const int end)
    \brief layer-specific forwarding function
    \param l the layer
    \param in the input images buffer
    \param out the output images buffer
    \param start starting position from a batch of inputs, default to 0
    \param end end position from a batch of inputs, default to 1
*/
/*! \fn void generic_forward_func(Layer* l, const vol_t** input, vol_t** output, const int start, const int end)
    \brief generic forward function which calls layer_wise specific forward function pointer and activation function pointer
    \param l the layer
    \param in the input images buffer
    \param out the output images buffer
    \param start starting position from a batch of inputs, default to 0
    \param end end position from a batch of inputs, default to 1
*/


#ifndef LAYER_H
#define LAYER_H

#include "data_structure.h"
#include "math_functions.h"
typedef int label_t;

/**
 * @brief layer types
 *
 * For enumerate different layer types
 */
typedef enum {
    CONVOLUTIONAL = 100, /**< convolutional layer */
    POOLING, /**< spatial max pooling layer */
    LRN,      /**< Normalization layer */
    AVE_POOLING, /**<spatial average pooling layer */
    SOFTMAX, /**< softmax layer */
    RELU,    /**< ReLU layer */
    FULLY_CONNECTED, /**< fully connected layer */
} LAYER_TYPE;


/**
 * @brief Forwarding layer definition
 *
 * Generic Layer definition, all other layers pointing to this
 */
typedef struct Layer{
    LAYER_TYPE type;
    
    int sx; /**<kernel width */
    int sy; /**<kernel height */

    int in_sx; /**<input image width */
    int in_sy; /**<input image height */
    int in_depth; /**<input image channel (or depth) */
    int num_inputs; /**<batch size */

    int stride; /**<stride number, used in convolutional and spatial pooling layer */
    int pad; /**<pad number, used in convolutional or spatial pooling layer */

    int out_depth; /**<output channel (or depth) */
    int out_sx;    /**<output image width */
    int out_sy;    /**<output image height */

    storage_t* es; /**<used to hold intermediate values in softmax layer */


    double alpha; /**<parameter for LRN layer */
    double beta;  /**<parameter for LRN layer */

    storage_t bias;
    vol_t* biases;  /**<used to hold biases */
    vol_t** filters; /**<used to hold weights in fully-connected layer */
    vol_t* filters_flat; /**<used to hold weights in convolutional layer */
    storage_t* col_inputs; /**<buffer space for matrix expansion in convolutional layer */
    void (*forward)(const struct Layer* l, const vol_t** in, vol_t** out, const int start, const int end); /**<function pointer to layer specific forwarding function */
    void (*activation)(const struct Layer* l, vol_t** output, const int start, const int end); /**<function pointer to layer specific activation function */
} Layer;

void generic_forward_func(Layer* l, const vol_t** input, vol_t** output, const int start, const int end){
    (*l->forward)(l, input, output, start, end);
    if(l->activation){
        (*l->activation)(l, output, start, end);
    }
}

#endif
