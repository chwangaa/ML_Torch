#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"
#include "setting.h"
#include <loki/lokilib.h>
typedef Layer fc_layer_t;


typedef struct fc_global_data_ {
  fc_layer_t* fc_layer;
  vol_t** input;
  vol_t** output;
  int start;
  int end;
  int cores;
} fc_global_data;

void fc_forward_Worker(const fc_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end, const int cores) {
  uint core = get_core_id() + 8*tile2int(get_tile_id());

  for (int j = start; j <= end; j++) {
    vol_t* V = in[j];
    vol_t* A = out[j];
    storage_t* inputs = V->w;
    for(int i=core; i<l->out_depth; i+=cores) {
      // for each output
      storage_t a = l->biases->w[i]; // initialize the accum
      storage_t* weights = l->filters[i]->w;
      for(int d=0;d<l->num_inputs; d++) {
        a = add(a, multiply(inputs[d], weights[d]));
      }
      A->w[i] = a;
    }
  }
}

void fc_forward_workerCore(const void* data) {
  // Add extra memory channel so we can load from both matrices in parallel.
  int core = get_core_id();
  
  fc_global_data* d = (fc_global_data*)data;
  fc_forward_Worker(d->fc_layer, d->input, d->output, d->start, d->end, d->cores);

  loki_sync(d->cores);
}


void fc_forward_MultiCore(const fc_layer_t* l, const vol_t** in, vol_t** out, const int start, const int end) {  
  const int cores = FC_LAYER_NUM_CORE;
  loki_init_default(cores, 0);

  /* construct a closure for the worker core */
  fc_global_data* data = malloc(sizeof(fc_global_data));
  data->fc_layer = l;
  data->input = in;
  data->output = out;
  data->start = start;
  data->end = end;
  data->cores = cores;
  
  distributed_func* config = malloc(sizeof(distributed_func));
  config->cores = cores;
  config->func = &fc_forward_workerCore;
  config->data = data;
  config->data_size = sizeof(fc_global_data);
  // start_energy_log();
  loki_execute(config);
  // stop_energy_log();  
}



fc_layer_t* make_fc_layer(int in_sx, int in_sy, int in_depth,
                          int num_neurons) {
  fc_layer_t* l = (fc_layer_t*)malloc(sizeof(fc_layer_t));

  // required
  l->in_depth = in_depth;
  l->in_sx = in_sx;
  l->in_sy = in_sy;
  l->out_depth = num_neurons;

  // computed
  l->num_inputs = l->in_sx * l->in_sy * l->in_depth;
  l->out_sx = 1;
  l->out_sy = 1;

  l->filters = (vol_t**)malloc(sizeof(vol_t*)*num_neurons);
  for (int i = 0; i < l->out_depth; i++) {
    l->filters[i] = make_vol(1, 1, l->num_inputs, 0);
  }

  l->bias = 0;
  l->biases = make_vol(1, 1, l->out_depth, l->bias);

  l->forward = &fc_forward_MultiCore;

  return l;
}

void fc_load_file(fc_layer_t* l, const char* fn) {
  FILE* fin = fopen(fn, "r");
  assert(fin);
  
  int num_inputs;
  int out_depth;
  int items_read;
  items_read = fscanf(fin, "%d %d", &num_inputs, &out_depth);
  assert(items_read == 2);
  assert(out_depth == l->out_depth);
  assert(num_inputs == l->num_inputs);

  for(int i = 0; i < l->out_depth; i++)
    for(int d = 0; d < l->num_inputs; d++) {
      double val;
      items_read = fscanf(fin, "%lf", &val);
      assert(items_read == 1);
      l->filters[i]->w[d] = readDouble(val);
    }

  for(int i = 0; i < l->out_depth; i++) {
    double val;
    items_read = fscanf(fin, "%lf", &val);
    assert(items_read == 1);
    l->biases->w[i] = readDouble(val);
  }

  fclose(fin);
}

void fc_load(fc_layer_t* l, const int* params, const weight_t* weights) {
  
  int num_inputs = params[0];
  int out_depth = params[1];
  assert(out_depth == l->out_depth);
  assert(num_inputs == l->num_inputs);

  int weight = 0;
  for(int i = 0; i < l->out_depth; i++)
    for(int d = 0; d < l->num_inputs; d++) {
      double val = weights[weight++];
      l->filters[i]->w[d] = readDouble(val);
    }

  for(int i = 0; i < l->out_depth; i++) {
    double val = weights[weight++];
    l->biases->w[i] = readDouble(val);
  }

}

#endif
