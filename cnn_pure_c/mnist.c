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
#include "convolutional_layer.h"
#include "fully_connected_layer.h"
#include "max_pooling_layer.h"
#include "soft_max_layer.h"
#include "network.h"
#include "util.h"

#include "models/mnist/model.h"

// Neural Network -------------------------------------------------------------
// Load the snapshot of the CNN we are going to run.
Network* construct_mnist_net() {
  Network* net = make_network(8);

  network_add(net, make_conv_layer(28, 28, 1, 5, 6, 1, 0));
  network_add(net, make_max_pool_layer(net->layers[0]->out_sx, net->layers[0]->out_sy, net->layers[0]->out_depth, 2, 2));
  network_add(net, make_conv_layer(net->layers[1]->out_sx, net->layers[1]->out_sy, net->layers[1]->out_depth, 5, 16, 1, 0));
  network_add(net, make_max_pool_layer(net->layers[2]->out_sx, net->layers[2]->out_sy, net->layers[2]->out_depth, 2, 2));
  network_add(net, make_fc_layer(net->layers[3]->out_sx, net->layers[3]->out_sy, net->layers[3]->out_depth, 120));
  network_add(net, make_relu_layer(net->layers[4]->out_sx, net->layers[4]->out_sy, net->layers[4]->out_depth));
  network_add(net, make_fc_layer(net->layers[5]->out_sx, net->layers[5]->out_sy, net->layers[5]->out_depth, 10));
  network_add(net, make_softmax_layer(net->layers[6]->out_sx, net->layers[6]->out_sy, net->layers[6]->out_depth));

  // load pre-trained weights
  conv_load(net->layers[0], mnist_conv1_params, mnist_conv1_data);
  conv_load(net->layers[2], mnist_conv2_params, mnist_conv2_data);
  fc_load(net->layers[4], mnist_ip1_params, mnist_ip1_data);
  fc_load(net->layers[6], mnist_ip2_params, mnist_ip2_data);
  fprintf(stderr, "loading data complete \n");
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
          int items_read = fscanf(fin, "%d", &val);
          assert(items_read == 1);
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
    int items_read = fscanf(fin, "%d", &val);
    assert(items_read == 1);
    label[i] = val;
  }

  fclose(fin);
  fprintf(stderr, "input batch loaded successfully \n");
}

int main(int argc, char** argv) {
  int NUM_PASSES = 100;
  if (argc < 3) {
    fprintf(stderr, "Usage: ./mnist <test|inference> [NUMBER_PASSES]\n");
    return 2;
  }
  else{
    NUM_PASSES = atoi(argv[2]);
  }
  Network* net = construct_mnist_net();
  initialize_network(net, 1);
  vol_t** input = (vol_t**)malloc(sizeof(vol_t*)*NUM_PASSES);
  label_t* labels = (label_t*)malloc(sizeof(label_t)*NUM_PASSES);
  load_mnist_data(input, labels, NUM_PASSES); 
  
  if (!strcmp(argv[1], "inference")) {
    net_predict_Multiple(net, input, NUM_PASSES);
    return 1;
  }
  
  if (!strcmp(argv[1], "test")){
    net_test(net, input, labels, NUM_PASSES);
  }
  free_network(net);
  for(int i = 0; i < NUM_PASSES; i++)
    free(input[i]);
  free(input);
  free(labels);

  return 0;
}
