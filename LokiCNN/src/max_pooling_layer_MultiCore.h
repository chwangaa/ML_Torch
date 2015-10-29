// MaxPool Layer -----------------------------------------------------------------
#ifndef MAX_POOLING_LAYER_H
#define MAX_POOLING_LAYER_H
#include <loki/lokilib.h>
#include "layer.h"
#include "setting.h"
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
    int index = 0;
    int Vsx = V->sx;
    int Vsy = V->sy;
    int lsx = l->sx;
    int lsy = l->sy;
    int stride = l -> stride;
    int pad = -l->pad;

    for(int d=core; d<l->out_depth; d+=cores) {
      int x = pad;
      int y = pad;
      for(int ax=0; ax<l->out_sx; x+=stride,ax++) {
        y = pad;
        for(int ay=0; ay<l->out_sy; y+=stride,ay++) {
  
          storage_t a = readDouble(-99999.0);
          for(int fx = 0 ; fx < lsx; fx++) {
            for(int fy = 0; fy < lsy; fy++) {
              int oy = y+fy;
              int ox = x+fx;
              if(oy<Vsy && ox>=0 && ox<Vsx) {
                storage_t v = get_vol(V, ox, oy, d);
                if(v > a) { a = v; }
              }
            }
          }
          // set_vol(A, ax, ay, d, a);
          A->w[index++] = a;
        }
      }
    }
  }
}

void max_pool_forward_workerCore(const void* data) {
  // Add extra memory channel so we can load from both matrices in parallel.
  int core = get_core_id();
  
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
  l->out_sx = ceil((l->in_sx + l->pad * 2 - l->sx) / l->stride + 1);
  l->out_sy = ceil((l->in_sy + l->pad * 2 - l->sy) / l->stride + 1);

  l->forward = &max_pool_forward_MultiCore;

  return l;
}

#endif