// MaxPool Layer -----------------------------------------------------------------
#ifndef MAX_POOLING_LAYER_H
#define MAX_POOLING_LAYER_H

#include "layer.h"
#include "util.h"
typedef Layer pool_layer_t;

typedef struct pool_data_ {
  pool_layer_t* pool_layer;
  vol_t** input;
  vol_t** output;
  int start;
  int end;
  int cores;
} pool_data;


__attribute__ ((noinline)) void max_pool_forward_Worker(const pool_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end, const int cores) {
  uint core = get_core_id() + 8*tile2int(get_tile_id());
  
  for (int i = start; i <= end; i++) {
    vol_t* V = in[i];
    vol_t* A = out[i];
        
    for(int d=core; d<l->out_depth; d+=cores) {
      fprintf(stderr, "core %d is computing level %d \n", core, d);
      int x = -l->pad;
      int y = -l->pad;
      for(int ax=0; ax<l->out_sx; x+=l->stride,ax++) {
        y = -l->pad;
        for(int ay=0; ay<l->out_sy; y+=l->stride,ay++) {
  
          double a = -99999;
          for(int fx=0;fx<l->sx;fx++) {
            for(int fy=0;fy<l->sy;fy++) {
              int oy = y+fy;
              int ox = x+fx;
              if(oy>=0 && oy<V->sy && ox>=0 && ox<V->sx) {
                double v = get_vol(V, ox, oy, d);
                if(v > a) { a = v; }
              }
            }
          }
          set_vol(A, ax, ay, d, a);
        }
      }
    }
  }
}

void max_pool_forward_workerCore(const void* data) {
  // Add extra memory channel so we can load from both matrices in parallel.
  int core = get_core_id();
  int addr = loki_mem_address(0, core, CH_REGISTER_3, GROUPSIZE_8, false, false, true, true);  
  set_channel_map(2, addr);
  
  pool_data* d = (pool_data*)data;
  max_pool_forward_Worker(d->pool_layer, d->input, d->output, d->start, d->end, d->cores);

  loki_sync(d->cores);
}


void max_pool_forward_MultiCore(const pool_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end) {  
  const int cores = MAX_POOLING_LAYER_NUM_CORE;
  loki_init_default(cores, 0);

  /* construct a closure for the worker core */
  pool_data* data = malloc(sizeof(pool_data));
  data->pool_layer = l;
  data->input = in;
  data->output = out;
  data->start = start;
  data->end = end;
  data->cores = cores;
  
  distributed_func* config = malloc(sizeof(distributed_func));
  config->cores = cores;
  config->func = &max_pool_forward_workerCore;
  config->data = data;
  config->data_size = sizeof(pool_data);
  // start_energy_log();
  loki_execute(config);
  // stop_energy_log();  
}

pool_layer_t* make_max_pool_layer(int in_sx, int in_sy, int in_depth,
                              int sx, int stride) {
  pool_layer_t* l = (pool_layer_t*)malloc(sizeof(pool_layer_t));

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

  l->forward = &max_pool_forward_MultiCore;
  return l;
}

#endif
