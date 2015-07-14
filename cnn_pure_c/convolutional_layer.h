// Convolutional Layer --------------------------------------------------------
#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "layer.h"

typedef Layer conv_layer_t;

void conv_forward_with_padding(conv_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
        
    const int V_sx = V->sx;
    const int V_sy = V->sy;
    const int xy_stride = l->stride;

    const int outx = l->out_sx;
    const int outy = l->out_sy;
  
    for(int d = 0; d < l->out_depth; d++) {
      const vol_t* f = l->filters[d];
      const storage_t bias = l->biases->w[d];
      const int width = f->sx;
      const int height = f->sy;
      const int depth = f->depth;
      


      // used to track the top-left of input matrix currently under computation
      const int reset_x = -l->pad;
      int x;
      int y = -l->pad;

      // used to track the output coordinate
      int ax = 0;
      int ay = 0;

      // used to accumulate matrix sum
      storage_t a; 

      /*
      start doing the computation
      the code is divided into 3 sections
      1): where y is padded lower than 0
      2): normal y
      3): where y is padded over the height
      */
      
      // first section
      for(; y < 0; y+=xy_stride, ay++){
        x = reset_x;
        for(ax=0; ax < outx; x += xy_stride, ax++) {
          
          a = 0.0;
          int index = 0;
          for(int fd=0;fd < depth; fd++) {
            for(int fx = x; fx < x + width; fx++) {
                
                for(int fy = y; fy < y + height; fy++, index++) {
                  
                  if(fy >= 0 && fx >=0 && fx < V_sx) {
                    a += f->w[index] * get_vol(V, fx, fy, fd);
                  }
              }
            }
          }
          a += bias;
          set_vol(A, ax, ay, d, a);
        }
      }


      /* second section is further divided into three sub sections
      2.1): where x is padded lower than 0
      2.2): normal x
      2.3): where x is padded over the width
      */
        // note down the reference locations first
        int reset_y = y;
        int reset_ay = ay;

        x = reset_x;
        
        // section 2.1
        for(ax=0; x < 0 && ax < outx; x += xy_stride, ax++) {
          ay = reset_ay;
          for(y=reset_y; ay < outy && y <= V_sy - height; y += xy_stride, ay++) {
            a = 0.0;
            int index = 0;
            for(int fd=0;fd < depth; fd++) {
              for(int fx = x; fx < x + width; fx++) {
                  
                  for(int fy = y; fy < y + height; fy++, index++) {
                    
                    if(fx >=0) {
                      a += f->w[index] * get_vol(V, fx, fy, fd);
                    }
                }
              }
            }
            a += bias;
            set_vol(A, ax, ay, d, a);
          }
        }

        // section 2.2
        for(; x < V_sx - width; x += xy_stride, ax++) {
          ay = reset_ay;
          for(y=reset_y; ay < outy && y <= V_sy - height; y += xy_stride, ay++) {
            a = 0.0;
            int index = 0;
            
            for(int fd=0;fd < depth; fd++) {
              for(int fx = x; fx < x + width; fx++) {
                for(int fy = y; fy < y + height; fy++, index++) {
                  a += f->w[index] * get_vol(V, fx, fy, fd);
                }
              }
            }

            a += bias;
            set_vol(A, ax, ay, d, a);
          }
        }

        // section 2.3
        for(; ax < outx; x += xy_stride, ax++) {
          ay = reset_ay;
          for(y=reset_y; ay < outy && y <= V_sy - height; y += xy_stride, ay++) {
            a = 0.0;
            int index = 0;
            for(int fd=0;fd < depth; fd++) {
              for(int fx = x; fx < x + width; fx++) {
                  for(int fy = y; fy < y + height; fy++, index++) {
                    if(fx < V_sx) {
                      a += f->w[index] * get_vol(V, fx, fy, fd);
                    }
                }
              }
            }
            a += bias;
            set_vol(A, ax, ay, d, a);
          }
        }


      // third section
      for(; ay < outy; y += xy_stride, ay++) {
        x = reset_x;
        for(ax=0; ax < outx; x += xy_stride, ax++) {
          
          a = 0.0;
          int index = 0;
          for(int fd=0;fd < depth; fd++) {
            for(int fx = x; fx < x + width; fx++) {
                for(int fy = y; fy < y + height; fy++, index++) {
                  if(fy >= 0 && fy < V_sy && fx >=0 && fx < V_sx) {
                    a += f->w[index] * get_vol(V, fx, fy, fd);
                  }
              }
            }
          }
          a += bias;
          set_vol(A, ax, ay, d, a);
        }
      }

    }
  }
}


void conv_forward_without_padding(conv_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
        
    const int V_sx = V->sx;
    const int V_sy = V->sy;
    const int xy_stride = l->stride;

    const int outx = l->out_sx;
    const int outy = l->out_sy;
  
    for(int d = 0; d < l->out_depth; d++) {

      const vol_t* f = l->filters[d];
      const storage_t bias = l->biases->w[d];
      const int width = f->sx;
      const int height = f->sy;
      const int depth = f->depth;
      


      // used to track the top-left of input matrix currently under computation
      const int reset_x = -l->pad;
      int x;
      int y = -l->pad;

      // used to track the output coordinate
      int ax = 0;
      int ay = 0;

      // used to accumulate matrix sum
      storage_t a;

      for(; ay < outy; y += xy_stride, ay++) {
        x = reset_x;
        for(ax=0; ax < outx; x += xy_stride, ax++) {
          
          a = 0.0;
          int index = 0;
          for(int fd=0;fd < depth; fd++) {
            for(int fx = x; fx < x + width; fx++) {
                for(int fy = y; fy < y + height; fy++, index++) {
                    a += f->w[index] * get_vol(V, fx, fy, fd);
              }
            }
          }
          a += bias;
          set_vol(A, ax, ay, d, a);
        }
      }
    }
  }
}


conv_layer_t* make_conv_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int filters, int stride, int pad) {
  conv_layer_t* l = (conv_layer_t*)malloc(sizeof(conv_layer_t));

  // required
  l->sx = sx;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
  l->out_depth = filters;
    
  // optional
  l->sy = l->sx;
  l->stride = stride;
  l->pad = pad;
  l->l1_decay_mul = 0.0;
  l->l2_decay_mul = 1.0;

  //
  assert(l->pad >=0);

  // computed
  l->out_sx = floor((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = floor((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*filters);
  for (int i = 0; i < filters; i++) {
    l->filters[i] = make_vol(l->sx, l->sy, l->in_depth, 0.0);
  }

  l->bias = 0.0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  if(l->pad != 0){
    l->forward = &conv_forward_with_padding;
  }
  else{
    l->forward = &conv_forward_without_padding;   
  }
  return l;
}

void conv_load_file(conv_layer_t* l, const char* fn) {
  FILE* fin = fopen(fn, "r");
  assert(fin != NULL);
  
  int sx, sy, depth, filters;
  sx = sy = depth = filters = 0;
  fscanf(fin, "%d %d %d %d", &sx, &sy, &depth, &filters);
  assert(sx == l->sx);
  assert(sy == l->sy);
  assert(depth == l->in_depth);
  assert(filters == l->out_depth);

  for(int d = 0; d < l->out_depth; d++)
    for (int z = 0; z < depth; z++)
      for (int x = 0; x < sx; x++)
        for (int y = 0; y < sy; y++){
          double val;
          fscanf(fin, "%lf", &val);
          // fprintf(stderr, "value read is %f \n", val);
          set_vol(l->filters[d], x, y, z, val);
        }

  // fprintf(stderr, "weights loaded correctly \n");

  for(int d = 0; d < l->out_depth; d++) {
    double val;
    fscanf(fin, "%lf", &val);
    set_vol(l->biases, 0, 0, d, val);
  }

  fclose(fin);
}

void conv_load(conv_layer_t* l, const int* params, const weight_t* weights) {  
  int sx, sy, depth, filters;
  sx = params[0]; sy = params[1]; depth = params[2]; filters = params[3];
  assert(sx == l->sx);
  assert(sy == l->sy);
  assert(depth == l->in_depth);
  assert(filters == l->out_depth);

  int i=0;
  for(int d = 0; d < l->out_depth; d++)
    for (int z = 0; z < depth; z++)
      for (int x = 0; x < sx; x++)
        for (int y = 0; y < sy; y++){
          weight_t val = weights[i++];
          // fprintf(stderr, "value read is %f \n", val);
          set_vol(l->filters[d], x, y, z, val);
        }

  // fprintf(stderr, "weights loaded correctly \n");

  for(int d = 0; d < l->out_depth; d++) {
    weight_t val = weights[i++];
    set_vol(l->biases, 0, 0, d, val);
  }

}

#endif
