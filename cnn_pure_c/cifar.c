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
#include "pooling_layer.h"
#include "soft_max_layer.h"
#include "network.h"
#include "util.h"
// Neural Network -------------------------------------------------------------
// Load the snapshot of the CNN we are going to run.
Network* construct_cifar_net() {
  fprintf(stderr, "Constructing Cifar Network \n");
  Network* net = make_network(11);

  network_add(net, make_conv_layer(32, 32, 3, 5, 16, 1, 2));
  network_add(net, make_relu_layer(net->layers[0]->out_sx, net->layers[0]->out_sy, net->layers[0]->out_depth));
  network_add(net, make_pool_layer(net->layers[1]->out_sx, net->layers[1]->out_sy, net->layers[1]->out_depth, 2, 2));
  network_add(net, make_conv_layer(net->layers[2]->out_sx, net->layers[2]->out_sy, net->layers[2]->out_depth, 5, 20, 1, 2));
  network_add(net, make_relu_layer(net->layers[3]->out_sx, net->layers[3]->out_sy, net->layers[3]->out_depth));
  network_add(net, make_pool_layer(net->layers[4]->out_sx, net->layers[4]->out_sy, net->layers[4]->out_depth, 2, 2));
  network_add(net, make_conv_layer(net->layers[5]->out_sx, net->layers[5]->out_sy, net->layers[5]->out_depth, 5, 20, 1, 2));
  network_add(net, make_relu_layer(net->layers[6]->out_sx, net->layers[6]->out_sy, net->layers[6]->out_depth));
  network_add(net, make_pool_layer(net->layers[7]->out_sx, net->layers[7]->out_sy, net->layers[7]->out_depth, 2, 2));
  network_add(net, make_fc_layer(net->layers[8]->out_sx, net->layers[8]->out_sy, net->layers[8]->out_depth, 10));
  network_add(net, make_softmax_layer(net->layers[9]->out_sx, net->layers[9]->out_sy, net->layers[9]->out_depth));

  // load pre-trained weights
  conv_load(net->layers[0], "models/cifar/layer1_conv.txt");
  conv_load(net->layers[3], "models/cifar/layer4_conv.txt");
  conv_load(net->layers[6], "models/cifar/layer7_conv.txt");
  fc_load(net->layers[9], "models/cifar/layer10_fc.txt");
  return net;
}

// Load an entire batch of images from the cifar10 data set (which is divided
// into 5 batches with 10,000 images each).
void load_cifar_data(vol_t** data, label_t* label, int size) {
  fprintf(stderr, "Loading Data \n");

  assert(size <= 10000);  // the size must be smaller than 10'000
  char fn[] = "data/cifar/data_batch_1.bin";
  FILE* fin = fopen(fn, "rb");
  assert(fin != NULL);

  // vol_t** batchdata = (vol_t**)malloc(sizeof(vol_t*) * size);

  for (int i = 0; i < size; i++) {
    uint8_t data_buffer[3073];
    assert(fread(data_buffer, 1, 3073, fin) == 3073);

    int outp = 0;

    data[i] = make_vol(32, 32, 3, 0.0);
    label[i] = data_buffer[outp++];

    for (int z = 0; z < 3; z++)
      for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++) {
          set_vol(data[i], x, y, z, ((double)data_buffer[outp++])/255.0-0.5);
        }
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

  Network* net = construct_cifar_net();
  initialize_network(net, 1);  
  vol_t** input = (vol_t**)malloc(sizeof(vol_t*)*NUM_PASSES);
  label_t* labels = (label_t*)malloc(sizeof(label_t)*NUM_PASSES);
  load_cifar_data(input, labels, NUM_PASSES);
  
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
  fprintf(stderr, "ERROR: Unknown command \n");

  return 2;
}
