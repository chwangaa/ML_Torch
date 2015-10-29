#include <lokicnn.h>

#include "../models/cifar/model.h"


// Embed the input data into the executable and access using these locations.
extern char _binary_data_cifar_data_batch_1_bin_start;
extern char _binary_data_cifar_data_batch_1_bin_end;

// Neural Network -------------------------------------------------------------
// Load the snapshot of the CNN we are going to run.
Network* construct_cifar_net() {
  fprintf(stderr, "Constructing Cifar Network \n");
  Network* net = make_network(11);

  network_add(net, make_conv_layer(32, 32, 3, 5, 16, 1, 2));
  network_add(net, make_relu_layer(net->layers[0]->out_sx, net->layers[0]->out_sy, net->layers[0]->out_depth));
  network_add(net, make_max_pool_layer(net->layers[1]->out_sx, net->layers[1]->out_sy, net->layers[1]->out_depth, 2, 2));
  network_add(net, make_conv_layer(net->layers[2]->out_sx, net->layers[2]->out_sy, net->layers[2]->out_depth, 5, 20, 1, 2));
  network_add(net, make_relu_layer(net->layers[3]->out_sx, net->layers[3]->out_sy, net->layers[3]->out_depth));
  network_add(net, make_max_pool_layer(net->layers[4]->out_sx, net->layers[4]->out_sy, net->layers[4]->out_depth, 2, 2));
  network_add(net, make_conv_layer(net->layers[5]->out_sx, net->layers[5]->out_sy, net->layers[5]->out_depth, 5, 20, 1, 2));
  network_add(net, make_relu_layer(net->layers[6]->out_sx, net->layers[6]->out_sy, net->layers[6]->out_depth));
  network_add(net, make_max_pool_layer(net->layers[7]->out_sx, net->layers[7]->out_sy, net->layers[7]->out_depth, 2, 2));
  network_add(net, make_fc_layer(net->layers[8]->out_sx, net->layers[8]->out_sy, net->layers[8]->out_depth, 10));
  network_add(net, make_softmax_layer(net->layers[9]->out_sx, net->layers[9]->out_sy, net->layers[9]->out_depth));

  // load pre-trained weights
  conv_load(net->layers[0], layer1_conv_params, layer1_conv_data);
  conv_load(net->layers[3], layer4_conv_params, layer4_conv_data);
  conv_load(net->layers[6], layer7_conv_params, layer7_conv_data);
  fc_load(net->layers[9], layer10_fc_params, layer10_fc_data);
  return net;
}

// Load an entire batch of images from the cifar10 data set (which is divided
// into 5 batches with 10,000 images each).
void load_cifar_data(vol_t** data, label_t* label, int size) {
  fprintf(stderr, "Loading Data \n");

  assert(size <= 10000);  // the size must be smaller than 10'000
  
  uint8_t* data_buffer = &_binary_data_cifar_data_batch_1_bin_start;
  int outp = 0;

  for (int i = 0; i < size; i++) {
    data[i] = make_vol(32, 32, 3, 0);
    label[i] = data_buffer[outp++];

    for (int z = 0; z < 3; z++)
      for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++) {
	         double val_d = data_buffer[outp++]/255.0 - 0.5;
	         storage_t val = readDouble(val_d);
           set_vol(data[i], x, y, z, val);
        }
  }

  fprintf(stderr, "input batch loaded successfully \n");
}

int main(int argc, char** argv) {
  int NUM_PASSES = 100;
  if (argc < 3) {
    fprintf(stderr, "Usage: ./mnist <test|summary> [NUMBER_PASSES]\n");
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
  
  if (!strcmp(argv[1], "test")){
    net_test(net, input, labels, NUM_PASSES);
  }
  else if(!strcmp(argv[1], "summary")){
    net_summary(net);
  }
  else{
    fprintf(stderr, "Usage: ./mnist <test|summary> [NUMBER_PASSES]\n");    
  }
  free_network(net);
  for(int i = 0; i < NUM_PASSES; i++)
    free(input[i]);
  free(input);
  free(labels);

  return 0;
}
