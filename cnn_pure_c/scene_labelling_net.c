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

// Model referenced in paper: http://delivery.acm.org/10.1145/2750000/2744788/a108-cavigelli.pdf?ip=131.111.184.18&id=2744788&acc=ACTIVE%20SERVICE&key=BF07A2EE685417C5%2E6CDC43D2A5950A53%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=693103990&CFTOKEN=72630065&__acm__=1436879082_abb335b0c6bff6ea2d573dafecbbe01a
// Used as a benchmark in Origami paper
Network* construct_scene_labeling_net() {
  Network* net = make_network(12);

  network_add(net, make_conv_layer(320, 240, 3, 7, 16, 1, 0));
  network_add(net, make_max_pool_layer(net->layers[0]->out_sx, net->layers[0]->out_sy, net->layers[0]->out_depth, 2, 2));
  network_add(net, make_relu_layer(net->layers[1]->out_sx, net->layers[1]->out_sy, net->layers[1]->out_depth));

  network_add(net, make_conv_layer(net->layers[2]->out_sx, net->layers[2]->out_sy, net->layers[2]->out_depth, 7, 64, 1, 0));
  network_add(net, make_max_pool_layer(net->layers[3]->out_sx, net->layers[3]->out_sy, net->layers[3]->out_depth, 2, 2));
  network_add(net, make_relu_layer(net->layers[4]->out_sx, net->layers[4]->out_sy, net->layers[4]->out_depth));

  network_add(net, make_conv_layer(net->layers[5]->out_sx, net->layers[5]->out_sy, net->layers[5]->out_depth, 7, 256, 1, 0));
  network_add(net, make_relu_layer(net->layers[6]->out_sx, net->layers[6]->out_sy, net->layers[6]->out_depth));

  network_add(net, make_fc_layer(net->layers[7]->out_sx, net->layers[7]->out_sy, net->layers[7]->out_depth, 64));
  network_add(net, make_relu_layer(net->layers[8]->out_sx, net->layers[8]->out_sy, net->layers[8]->out_depth));
  network_add(net, make_fc_layer(net->layers[9]->out_sx, net->layers[9]->out_sy, net->layers[9]->out_depth, 8));
  network_add(net, make_softmax_layer(net->layers[10]->out_sx, net->layers[10]->out_sy, net->layers[10]->out_depth));

  return net;
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
  Network* net = construct_scene_labeling_net();
  initialize_network(net, 1);
  
  if (!strcmp(argv[1], "inference")) {
    net_measure_inference_time(net);
    return 1;
  }
  
  if (!strcmp(argv[1], "test")){
    // net_test(net, input, labels, NUM_PASSES);
  }
  free_network(net);


  return 2;
}
