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

void net_classify_cats(Network* net, vol_t** input, int n) {
  for (int i = 0; i < n; i++) {
    copy_vol(net->buffer[0][0], input[i]);    // everytime, set the input at data_layer
    net_forward(net);      // run_forward
  }
}

// Load the snapshot of the CNN we are going to run.
Network* construct_cifar_net() {
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
vol_t** load_cifar_data(int size) {

  char fn[] = "data/cifar/data_batch_1.bin";
  FILE* fin = fopen(fn, "rb");
  assert(fin != NULL);

  vol_t** batchdata = (vol_t**)malloc(sizeof(vol_t*) * size);

  for (int i = 0; i < size; i++) {
    batchdata[i] = make_vol(32, 32, 3, 0.0);

    uint8_t data[3073];
    assert(fread(data, 1, 3073, fin) == 3073);

    int outp = 1;
    for (int z = 0; z < 3; z++)
      for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++) {
          set_vol(batchdata[i], x, y, z, ((double)data[outp++])/255.0-0.5);
        }
  }

  fclose(fin);
  fprintf(stderr, "input batch loaded successfully \n");

  return batchdata;
}


int performance_measure(int num_samples) {

  fprintf(stderr, "RUN INFERENCE ALGORITHM ON %d PICTURES...\n", num_samples);

  fprintf(stderr, "Constructing Cifar Network \n");
  Network* net = construct_cifar_net();
  initialize_network(net, 1);

  fprintf(stderr, "Loading Data \n");
  // vol_t** input = (vol_t**)malloc(sizeof(vol_t*)*n);
  vol_t** input = load_cifar_data(num_samples);

  fprintf(stderr, "Running classification...\n");
  uint64_t start_time = timestamp_us(); 
  net_classify_cats(net, input, num_samples);
  uint64_t end_time = timestamp_us();

  double dt = (double)(end_time-start_time) / 1000.0;
  fprintf(stderr, "TIME: %.2lf ms\n", dt);
  fprintf(stderr, "\nTime/Image %.2lf ms \n\n", (dt/ (double)num_samples));

  free_network(net);
  free(input);

  return 0;
}

/*
 * The actual main function.
 */

int main(int argc, char** argv) {
  int BENCHMARK_SIZE = 1000;
  if (argc < 2) {
    printf("Using DEFAULT BENCHMARK OF %d \n", BENCHMARK_SIZE);
  }
  else{
    BENCHMARK_SIZE = atoi(argv[1]);
  }
  return performance_measure(BENCHMARK_SIZE);  
  fprintf(stderr, "ERROR: Unknown command\n");

  return 2;
}
