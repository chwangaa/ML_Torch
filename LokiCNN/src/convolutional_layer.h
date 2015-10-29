/*! \file convolutional_layer.h
    \brief GEMM version of convolutional layer
*/
// Convolutional Layer --------------------------------------------------------
#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "layer.h"
#include "math_functions.h"

typedef Layer conv_layer_t;

/*! 
    \brief conv layer forward routine
    \param l the layer, internally defines convolution parameters like kernel size, padding size, etc.
    \param in the input images
    \param out the ouput images
    \param start the index in the input batch from which to perform forwarding
    \param end the index in the input batch from which to stop perform forwarding

    start and end helps to define the batch size, i.e. batch_size = end-start. They are useful as one can flexibly decides
    whether to load a lot of batches into main memory and use indices to define batch size. And even variable batch_size is 
    possible with changing values of start and end
*/
void conv_forward(const conv_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end) {
  for (int i = start; i <= end; i++) {
    
    // M is the number of outputs
    unsigned int M = l->out_depth;
    // K is the size of weights for one output
    unsigned int K = l->sx * l->sx * l->in_depth;
    // N is the size of one output image
    unsigned int N = l->out_sx * l->out_sy;

    // col_inputs is the buffer for input matrix A
    storage_t* col_inputs = l->col_inputs;

    // convert input image to a huge matrix, store in A
    function_summary(im2col, in[i]->w, l->in_depth, l->in_sx, l->in_sy, l->sx, l->stride, l->pad, col_inputs);    

    // assign the weight
    assign_bias(M, N, l->biases->w, out[i]->w);

    // C += A*B
    // note function_summary macro monitors the cycle count
    function_summary(cblas_gemm, 
                     M, N, K, 
                     l->filters_flat->w, K,
                     col_inputs, N,
                     out[i]->w, N);
  }
}

/*! 
    \brief constructor of convolutional layer
    \param in_sx the width of input image
    \param in_sy the height of input image
    \param in_depth the channel number of input image
    \param sx the width of convolution kernel, note the kernel must be square
    \param filters the number of output images, this means the total weight size is filters*sx*sx
    \param stride the number of pixels between consecutive convolutional kernel movement, usually 1
    \param pad the number of 0 paddings to add to the input image before applying convolutions
*/
conv_layer_t* make_conv_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int filters, int stride, int pad) {
  assert(pad >=0);
  assert(in_sx == in_sy);

  conv_layer_t* l = (conv_layer_t*)malloc(sizeof(conv_layer_t));

  l->type = CONVOLUTIONAL;
  // required
  l->sx = sx;             // kernel size
  l->in_depth = in_depth; // input depth
  l->in_sx = in_sx;       // input width
  l->in_sy = in_sy;       // input height
  l->out_depth = filters; // output depth
  
  // optional
  l->sy = l->sx;          /// for LokiCNN, it is required that kernel width == kernel height
  l->stride = stride;     // stride
  l->pad = pad;           // pad

  // computed
  l->out_sx = floor((in_sx + pad * 2 - sx) / stride + 1);
  l->out_sy = floor((in_sy + pad * 2 - sx) / stride + 1);
  l->col_inputs = (storage_t*)malloc(sizeof(storage_t)*sx*sx*l->out_sx*l->out_sy*in_depth);

  /// the images are stored in one dimension flat arrays
  l->filters_flat = make_vol(sx*sx*in_depth, filters, 1, 0);

  l->biases = make_vol(1, 1, l->out_depth, 0);

  l->forward = &conv_forward;
  
  return l;
}


/*! 
    \brief default convolutional layer weight loader
    \param l pointer to the convolutional layer
    \param params specifiers of convolutional layer parameters, used to track whether the weight file matches the convolutional layer's requirement
    \param weights pointer to the starting point of weight array

    this is just one way of loading values, one can easily define its own loader according to specific needs
    for loading from weights trained by caffe, this is an easy approach
*/
void conv_load(conv_layer_t* l, const int* params, const weight_t* weights) {  
  int sx, sy, depth, filters;
  sx = params[0]; sy = params[1]; depth = params[2]; filters = params[3];
  assert(sx == l->sx);
  assert(sy == l->sy);
  assert(depth == l->in_depth);
  assert(filters == l->out_depth);

  int i=0;
  int index = 0;
  for(int d = 0; d < l->out_depth; d++){
    for (int z = 0; z < depth; z++)
      for (int x = 0; x < sx; x++)
        for (int y = 0; y < sy; y++){
          weight_t val = weights[i++];

          l->filters_flat->w[index++] = readDouble(val);
        }
  }

  for(int d = 0; d < l->out_depth; d++) {
    weight_t val = weights[i++];
    set_vol(l->biases, 0, 0, d, readDouble(val));
  }
}

#endif