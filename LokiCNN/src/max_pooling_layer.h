/*! \file max_pooling_layer.h
    \brief max pooling layer
*/
// MaxPool Layer -----------------------------------------------------------------
#ifndef MAX_POOLING_LAYER_H
#define MAX_POOLING_LAYER_H

#include "layer.h"
typedef Layer pool_layer_t;

/*! 
    \brief Spatial Max Pooling layer forward routine
    \param l the pooling layer pointer
    \param in the input images
    \param out the ouput images
    \param start the index in the input batch from which to perform forwarding
    \param end the index in the input batch from which to stop perform forwarding
*/
void max_pool_forward(const pool_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end) {
  for (int i = start; i <= end; i++) {
    const vol_t* V = in[i];
    vol_t* A = out[i];
    int index = 0;
    int Vsx = V->sx;
    int Vsy = V->sy;
    int lsx = l->sx;
    int lsy = l->sy;
    int stride = l -> stride;
    for(int d=0;d<l->out_depth;d++) {
      int x = -l->pad;
      int y = -l->pad;
      for(int ax=0; ax < l->out_sx; x += stride,ax++) {
        y = -l->pad;
        for(int ay=0; ay < l->out_sy; y += stride,ay++) {
  
          // usually the values are in magnitude of -2 to 2, so any minimum value is acceptable as absolute minimum
          storage_t a = readDouble(-99999.0);
          
          // iterate over the kernel and find the maximum
          for(int fx=0; fx < lsx;fx++) {
            for(int fy=0; fy < lsy;fy++) {
              int oy = y + fy;
              int ox = x + fx;
              if(oy >= 0 && oy < Vsy && ox >= 0 && ox < Vsx) {
                storage_t v = get_vol(V, ox, oy, d);
                if(v > a) { a = v; }
              }
            }
          }
          
          A->w[index++] = a;
        }
      }
    }
  }
}

/*! 
    \brief constructor of max_pooling layer
    \param in_sx the width of input image
    \param in_sy the height of input image
    \param in_depth the channel number of input image
    \param sx the width of pooling kernel, note the kernel must be square
    \param stride the number of pixels between consecutive convolutional kernel movement, usually 1

  in this simple implementation, we assume the max pooling layer has padding be 0, although one can simply insert one extra parameter
  and the forward layer already supports none-zero padding.
  The output dimension is calculated as following:

  out_sx = floor((l->in_sx + pad * 2 - sx) / stride + 1);
  
  out_sy = floor((l->in_sy + pad * 2 - sy) / stride + 1);

  usually, the output size is reduced by a factor of the kernel size
*/
pool_layer_t* make_max_pool_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int stride) {
  pool_layer_t* l = (pool_layer_t*)malloc(sizeof(pool_layer_t));
  l->type = POOLING;

  // required
  l->sx = sx; /*! it is required that the kernel height and kernel width are the same in LokiCNN */
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
  l->forward = &max_pool_forward;

  return l;
}

#endif
