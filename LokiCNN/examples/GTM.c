#include "lokicnn.h"
#include "../models/gtsrb/model.h"

// Embed the input data into the executable and access using these locations.
//extern char _binary_gtsrb_data_bin_start;
//extern char _binary_gtsrb_data_bin_end;

// Neural Network -------------------------------------------------------------
// Load the snapshot of the CNN we are going to run.
Network* construct_gtsrb_net() {
  fprintf(stderr, "Constructing GTSRB Network \n");
  Network* net = make_network(12);

  network_add(net, make_conv_layer(48, 48, 3, 3, 100, 1, 0));
  network_add(net, make_relu_layer(net->layers[0]->out_sx, net->layers[0]->out_sy, net->layers[0]->out_depth));
  network_add(net, make_max_pool_layer(net->layers[1]->out_sx, net->layers[1]->out_sy, net->layers[1]->out_depth, 2, 2));
  network_add(net, make_conv_layer(net->layers[2]->out_sx, net->layers[2]->out_sy, net->layers[2]->out_depth, 4, 150, 1, 0));
  network_add(net, make_relu_layer(net->layers[3]->out_sx, net->layers[3]->out_sy, net->layers[3]->out_depth));
  network_add(net, make_max_pool_layer(net->layers[4]->out_sx, net->layers[4]->out_sy, net->layers[4]->out_depth, 2, 2));
  network_add(net, make_conv_layer(net->layers[5]->out_sx, net->layers[5]->out_sy, net->layers[5]->out_depth, 3, 250, 1, 0));
  network_add(net, make_relu_layer(net->layers[6]->out_sx, net->layers[6]->out_sy, net->layers[6]->out_depth));
  network_add(net, make_max_pool_layer(net->layers[7]->out_sx, net->layers[7]->out_sy, net->layers[7]->out_depth, 2, 2));
  network_add(net, make_fc_layer(net->layers[8]->out_sx, net->layers[8]->out_sy, net->layers[8]->out_depth, 200));
  network_add(net, make_fc_layer(net->layers[9]->out_sx, net->layers[9]->out_sy, net->layers[9]->out_depth, 43));
  network_add(net, make_softmax_layer(net->layers[10]->out_sx, net->layers[10]->out_sy, net->layers[10]->out_depth));

  // load pre-trained weights
  conv_load(net->layers[0], conv1_params, conv1_data);
  conv_load(net->layers[3], conv2_params, conv2_data);
  conv_load(net->layers[6], conv3_params, conv3_data);
  fc_load(net->layers[9], ip1_params, ip1_data);
  fc_load(net->layers[10], ip2_params, ip2_data);
  return net;
}

// Load an entire batch of images from the gtsrb data set (which is divided
// into 5 batches with 10,000 images each).
void load_gtsrb_data(vol_t** data, label_t* label, int size) {
  fprintf(stderr, "Loading Data \n");

  assert(size <= 7830);  // the size must be smaller than 7830
  
  FILE *meanf = fopen("data/gtsrb/mean.binaryproto","rb");
  assert(meanf);
  float mean_buffer[6912];
  FILE *dataf = fopen("data/gtsrb/test.bin","rb");
  assert(dataf);
  uint8_t data_buffer[6913];
  assert(fread(mean_buffer,1,12,meanf) == 12);
  assert(fread(mean_buffer,sizeof(float),6912,meanf) == 6912);

  fclose(meanf);

  for (int i = 0; i < size; i++) {
    int outp = 0;
    data[i] = make_vol(48, 48, 3, 0.0);

    assert(fread(data_buffer,sizeof(uint8_t),6913,dataf) == 6913);
    label[i] = data_buffer[0];
    storage_t val;
    for (int z = 0; z < 3; z++)
      for (int y = 0; y < 48; y++)
        for (int x = 0; x < 48; x++) {
          double val_d = data_buffer[outp+1]-mean_buffer[outp];
          val = readDouble(val_d);
          set_vol(data[i], y, x, z, val);
          outp++;
        }
  }
  
  fclose(dataf);

  fprintf(stderr, "input batch loaded successfully \n");
}

int main(int argc, char** argv) {
  int NUM_PASSES = 100;
  if (argc < 3) {
    fprintf(stderr, "Usage: ./gtsrb <test|summary> [NUMBER_PASSES]\n");
    return 2;
  }
  else{
    NUM_PASSES = atoi(argv[2]);
  }

  Network* net = construct_gtsrb_net();
  initialize_network(net, 1);  
  if(!strcmp(argv[1], "summary")){
    net_summary(net);
  }
  else if (!strcmp(argv[1], "test")){
    vol_t** input = (vol_t**)malloc(sizeof(vol_t*)*NUM_PASSES);
    label_t* labels = (label_t*)malloc(sizeof(label_t)*NUM_PASSES);
    load_gtsrb_data(input, labels, NUM_PASSES);
    net_test(net, input, labels, NUM_PASSES);
    for(int i = 0; i < NUM_PASSES; i++)
      free(input[i]);
    free(input);
    free(labels);
  }
  else{
    fprintf(stderr, "Usage: ./gtsrb <test|summary> [NUMBER_PASSES]\n");
  }
  free_network(net);
}
