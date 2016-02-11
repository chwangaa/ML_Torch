#include "lokicnn.h"
#include "../models/mnist/model.h"

// Embed the input data into the executable and access using these locations.
extern char _binary_data_mnist_data_bin_start;
extern char _binary_data_mnist_data_bin_end;
extern char _binary_data_mnist_labels_bin_start;
extern char _binary_data_mnist_labels_bin_end;

// Neural Network -------------------------------------------------------------
// Load the snapshot of the CNN we are going to run.
Network* construct_mnist_net() {
  fprintf(stderr, "construction of MNIST NET \n");
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
  fprintf(stderr, "loading weights \n");
  conv_load(net->layers[0], mnist_conv1_params, mnist_conv1_data);
  conv_load(net->layers[2], mnist_conv2_params, mnist_conv2_data);
  fc_load(net->layers[4], mnist_ip1_params, mnist_ip1_data);
  fc_load(net->layers[6], mnist_ip2_params, mnist_ip2_data);

  fprintf(stderr, "loading weight complete \n");

  return net;
}


void load_mnist_data(vol_t** data, label_t* label, int size) {

  assert(size <= 10000);  // the size must be smaller than 10'000
  char* data_buffer = &_binary_data_mnist_data_bin_start;
  int outp = 0;
    
  for (int i = 0; i < size; i++) {

    data[i] = make_vol(28, 28, 1, 0);

    for (int z = 0; z < 1; z++)
      for (int x = 0; x < 28; x++)
        for (int y = 0; y < 28; y++) {
          unsigned char val = data_buffer[outp++];
          double val_d = ((double)val)/256;
          set_vol(data[i], x, y, z, readDouble(val_d));
        }
  }

  char* label_buffer = &_binary_data_mnist_labels_bin_start;
  outp = 0;

  for(int i = 0; i < size; i++){
    unsigned char val = label_buffer[outp++];
    label[i] = val;
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
  Network* net = construct_mnist_net();
  initialize_network(net, 1);
  
  vol_t** input = (vol_t**)malloc(sizeof(vol_t*)*NUM_PASSES);
  label_t* labels = (label_t*)malloc(sizeof(label_t)*NUM_PASSES);
  load_mnist_data(input, labels, NUM_PASSES); 
  
  if (!strcmp(argv[1], "test")){
    net_test(net, input, labels, NUM_PASSES);
  }
  else if(!strcmp(argv[1], "summary")){
    net_summary(net);
  }
  else{
    fprintf(stderr, "Usage: ./mnist <test|summary> [NUMBER_PASSES]\n");
  }
  // print_vol(net->buffer[2][0], 0);
  free_network(net);

  
  for(int i = 0; i < NUM_PASSES; i++)
    free(input[i]);
  free(input);
  free(labels);
  
  return 0;
}
