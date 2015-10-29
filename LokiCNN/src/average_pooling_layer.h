/*! \file average_pooling_layer.h
    \brief average pooling layer
*/
// AvgPool Layer -----------------------------------------------------------------
#ifndef AVERAGE_POOLING_LAYER_H
#define AVERAGE_POOLING_LAYER_H

#include "layer.h"

typedef Layer ave_pool_layer_t;

/*! 
    \brief average pooling layer forward routine
    \param l the layer
    \param in the input images
    \param out the ouput images
    \param start the index in the input batch from which to perform forwarding
    \param end the index in the input batch from which to stop perform forwarding

*/
void ave_pool_forward(ave_pool_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
        
    const int KERNEL_SIZE = l->sx * l->sy;
    int Vsx = V->sx;
    int Vsy = V->sy;
    int lsx = l->sx;
    int lsy = l->sy;
    for(int d=0;d<l->out_depth;d++) {
      int x = -l->pad;
      int y = -l->pad;
      for(int ax=0; ax<l->out_sx; x+=l->stride,ax++) {
        y = -l->pad;
        for(int ay=0; ay<l->out_sy; y+=l->stride,ay++) {
  
          storage_t accum = 0;
          int missed = 0;

          for(int fy=0; fy < lsy; fy++) {
            for(int fx=0; fx < lsx; fx++) {
              int oy = y + fy;
              int ox = x + fx;
              if(oy >= 0 && oy < Vsy && ox >= 0 && ox < Vsx) {
                accum += get_vol(V,ox,oy,d);
              } else {
                missed++;
              }
            }
          }
          accum /= KERNEL_SIZE - missed;
          set_vol(A, ax, ay, d, accum);
        }
      }
    }
  }
}


/*! 
    \brief constructor of average pooling layer
    \param in_sx the width of input image
    \param in_sy the height of input image
    \param in_depth the channel number of input image
    \param sx the width of pooling kernel, note the kernel must be square
    \param stride the number of pixels between consecutive convolutional kernel movement, usually 1

  in this simple implementation, we assume the pooling layer has padding be 0, although one can simply insert one extra parameter
  and the forward layer already supports none-zero padding.
  The output dimension is calculated as following:

  out_sx = floor((l->in_sx + pad * 2 - sx) / stride + 1);
  
  out_sy = floor((l->in_sy + pad * 2 - sy) / stride + 1);

  usually, the output size is reduced by a factor of the kernel size
*/
ave_pool_layer_t* make_average_pool_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int stride) {
  ave_pool_layer_t* l = (ave_pool_layer_t*)malloc(sizeof(ave_pool_layer_t));
  l->type = AVE_POOLING;

  // required
  l->sx = sx;
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
  l->forward = &ave_pool_forward;

  return l;
}

#endif