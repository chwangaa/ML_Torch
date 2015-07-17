#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>

#include "data_structure.h"
#include "ReLU_layer.h"

#ifdef LOKI
  #include "convolutional_layer_MultiCore.h"
  #include "max_pooling_layer_MultiCore.h"
#else
  #include "convolutional_layer.h"
  #include "max_pooling_layer.h"
#endif

#include "fully_connected_layer.h"
#include "soft_max_layer.h"
#include "network.h"
#include "util.h"
#include "../models/mnist/model.h"

/*
  test for when both conv layer is parallized by MultiCore
*/

Network* construct_mnist_net() {
  Network* net = make_network(1);

  network_add(net, make_max_pool_layer(28, 28, 1, 2, 2));
  return net;
}


void load_mnist_data(vol_t** data, label_t* label, int size) {

  assert(size <= 10000);  // the size must be smaller than 10'000
  char fn[] = "data/mnist/data.txt";
  FILE* fin = fopen(fn, "rb");
  assert(fin != NULL);

  for (int i = 0; i < size; i++) {
    int outp = 0;

    data[i] = make_vol(28, 28, 1, 0.0);

    for (int z = 0; z < 1; z++)
      for (int x = 0; x < 28; x++)
        for (int y = 0; y < 28; y++) {
          int val;
          fscanf(fin, "%d", &val);
          set_vol(data[i], x, y, z, ((storage_t)val)/256);
          // fprintf(stderr, "the data read is %d \n", val);
        }
  }
  fclose(fin);

  char fn2[] = "data/mnist/labels.txt";
  fin = fopen(fn2, "rb");
  assert(fin != NULL);

  for(int i = 0; i < size; i++){
    label_t val;
    fscanf(fin, "%d", &val);
    label[i] = val;
  }

  fclose(fin);
  fprintf(stderr, "input batch loaded successfully \n");
}

int main(int argc, char** argv) {
  int NUM_PASSES = 1;
  Network* net = construct_mnist_net();
  initialize_network(net, 1);
  vol_t** input = (vol_t**)malloc(sizeof(vol_t*)*NUM_PASSES);
  label_t* labels = (label_t*)malloc(sizeof(label_t)*NUM_PASSES);
  load_mnist_data(input, labels, NUM_PASSES);
  

  copy_vol(net->buffer[0][0], input[0]);
  net_forward(net);
  print_vol(net->buffer[0][0], 0);
  print_vol(net->buffer[1][0], 0);

  free_network(net);
  free(labels);
  for(int i = 0; i < NUM_PASSES; i++){
    free(input[i]);
  }
  free(input);
  return 1;
}
