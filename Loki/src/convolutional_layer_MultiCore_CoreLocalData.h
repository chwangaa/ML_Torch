// Convolutional Layer --------------------------------------------------------
#ifndef CONVOLUTIONAL_LAYER_MC_H
#define CONVOLUTIONAL_LAYER_MC_H

#include "layer.h"
#include <loki/lokilib.h>
#include "util.h"

typedef Layer conv_layer_t;

typedef struct global_data_ {
  conv_layer_t* conv_layer;
  vol_t** input;
  vol_t** output;
  int start;
  int end;
  int cores;
} global_data;

/* MultiCore Worker */

__attribute__ ((noinline)) void conv_forward_Worker(const conv_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end, const int cores) {
  uint core = get_core_id() + 8*tile2int(get_tile_id());

  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
    const storage_t* inputs = V->w;
    const int V_sx = V->sx;
    const int V_sy = V->sy;
    const int Area = V_sx * V_sy;
    const int xy_stride = l->stride;

    const int outx = l->out_sx;
    const int outy = l->out_sy;

    for(int d = core; d < l->out_depth; d+=cores) {
      const vol_t* f = l->filters[d];
      const storage_t bias = l->biases->w[d];
      // storage_t* weights = f->w;
      const int width = f->sx;
      const int height = f->sy;
      const int depth = f->depth;
      int index_p = 0;  
      storage_t weights[height*width*depth];
      for(int depth_index = 0; depth_index < depth; depth_index++){
        for(int width_index = 0; width_index < width; width_index++){
          for(int height_index = 0; height_index < height; height_index++){
            weights[index_p++] = get_vol(f, width_index, height_index, depth_index);
          }
        }
      }
      int x = -l->pad;
      int y = -l->pad;
      
      int ax = 0;
      int ay = 0;
      for(; ax < outx; x += xy_stride, ax++) {
        y = -l->pad;
        int output_index = (A->sx * d + ax) * A->sy;
        
        for(ay=0; ay < outy; y += xy_stride, ay++) {

          storage_t a = bias;
          int index = 0;
          int input_index = V_sy * x + y;          
          int temp_index_1;
          int temp_index_2;
          
          for(int fd=0, temp_index_1 = y + V_sy*x;fd < depth; fd++, temp_index_1 += Area) {
            
            for(int fx = x, temp_index_2 = temp_index_1; fx < x + width; fx++, temp_index_2 += V_sy) {
                for(int fy = y, input_index = temp_index_2; fy < y + height; fy++) {
                    a += weights[index++] * inputs[input_index++];
              }
            }
          }

          
        // set_vol(A, ax, ay, d, a);
        A->w[output_index++] = a;
        }
      }
    }
  }
}

void conv_forward_workerCore(const void* data) {
  // Add extra memory channel so we can load from both matrices in parallel.
  int core = get_core_id();
  int addr = loki_mem_address(0, core, CH_REGISTER_3, GROUPSIZE_8, false, false, true, true);  
  set_channel_map(2, addr);
  
  global_data* d = (global_data*)data;
  conv_forward_Worker(d->conv_layer, d->input, d->output, d->start, d->end, d->cores);

  loki_sync(d->cores);
}


void conv_forward_MultiCore(const conv_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end) {  
  const int cores = CONV_LAYER_NUM_CORE;
  loki_init_default(cores, 0);

  /* construct a closure for the worker core */
  global_data* data = malloc(sizeof(global_data));
  data->conv_layer = l;
  data->input = in;
  data->output = out;
  data->start = start;
  data->end = end;
  data->cores = cores;
  
  distributed_func* config = malloc(sizeof(distributed_func));
  config->cores = cores;
  config->func = &conv_forward_workerCore;
  config->data = data;
  config->data_size = sizeof(global_data);
  // start_energy_log();
  loki_execute(config);
  // stop_energy_log();  
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

  l->forward = &conv_forward_MultiCore;
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
