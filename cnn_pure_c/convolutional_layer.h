// Convolutional Layer --------------------------------------------------------
#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "layer.h"

typedef Layer conv_layer_t;

void conv_forward(conv_layer_t* l, vol_t** in, vol_t** out, int start, int end) {
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
        
    int V_sx = V->sx;
    int V_sy = V->sy;
    int xy_stride = l->stride;
  
    for(int d = 0; d < l->out_depth; d++) {
      vol_t* f = l->filters[d];
      int x = -l->pad;
      int y = -l->pad;
      for(int ay = 0; ay < l->out_sy; y += xy_stride, ay++) {
        x = -l->pad;
        for(int ax=0; ax < l->out_sx; x += xy_stride, ax++) {
          double a = 0.0;
          
          int index_f = 0;
          int index_v = x * (V->sy) + y;
          
          for(int fd = 0; fd < f->depth; fd++){
            for(int fx = 0, ox = x; fx < f->sx; fx++, ox++){
              index_v = ((V->sx * fd) +ox)*(V->sy) + y;
              for(int fy = 0, oy = y; fy < f->sy; fy++, index_v++, index_f++){
                if(oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx){
                  a+= f->w[index_f] * V->w[index_v];
                }
              }
            }
          }
          // for(int fy = 0; fy < f->sy; fy++) {
          //   int oy = y + fy;
            
          //   for(int fx = 0; fx < f->sx; fx++) {
          //     int ox = x + fx;
          //     if(oy >= 0 && oy < V_sy && ox >=0 && ox < V_sx) {
                
          //       for(int fd=0;fd < f->depth; fd++){
          //         a += get_vol(f, fx, fy, fd) * get_vol(V, ox, oy, fd);
          //         // a += f->w[((f->sx * fy)+fx)*f->depth+fd] * V->w[((V_sx * oy)+ox)*V->depth+fd];
          //       }
          //     }
          //   }
          // }
          a += l->biases->w[d];
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
  l->out_depth = filters;
  l->sx = sx;
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
    
  // optional
  l->sy = l->sx;
  l->stride = stride;
  l->pad = pad;
  l->l1_decay_mul = 0.0;
  l->l2_decay_mul = 1.0;

  // computed
  l->out_sx = floor((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = floor((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*filters);
  for (int i = 0; i < filters; i++) {
    l->filters[i] = make_vol(l->sx, l->sy, l->in_depth, 0.0);
  }

  l->bias = 0.0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  l->forward = &conv_forward;
  return l;
}

void conv_load(conv_layer_t* l, const char* fn) {
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

#endif