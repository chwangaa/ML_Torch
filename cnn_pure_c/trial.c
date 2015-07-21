#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#include "data_structure.h"
#include "ReLU_layer.h"
#include "convolutional_layer.h"
#include "fully_connected_layer.h"
#include "max_pooling_layer.h"
#include "soft_max_layer.h"
#include "network.h"
#include "util.h"

#include "models/cifar/model.h"

// Neural Network -------------------------------------------------------------
// Load the snapshot of the CNN we are going to run.
clock_t start, end;
vol_t *null_vol;

Network* construct_net(int x, int y, int in_d, int k, int d, int s) {
  Network* net = make_network(1);

  network_add(net, make_conv_layer(x, y, in_d, k, d, s, k/2));

  return net;
}

void time_net (int xy, int in_d, int k, int d, int s,
    int NUM_PASSES, FILE *out, int val) {
  Network* net = construct_net(xy,xy,in_d,k,d,s);
  initialize_network(net, 1);
  start = clock();
  for(int i=0; i<NUM_PASSES; i++) {
    copy_vol(net->buffer[0][0], null_vol);
    net_forward(net);
  }
  end = clock();
  fprintf(out,"%d_%d_%d_%d_%d,%d,%f\n",xy,in_d,k,d,s,val,
         (double)(end-start)*1000.0/CLOCKS_PER_SEC);
  free_network(net);
}

int main(int argc, char** argv) {
  null_vol = make_vol(32,32,32,(storage_t)0);
  int NUM_PASSES = 100;
  char mode = 'A';
  int opt;
  optopt = '_';
  while((opt = getopt(argc, argv, "c:d:")) != -1) {
    if(opt == 'c') {
      NUM_PASSES = atoi(optarg);
    }
    else if (opt == 'd') {
      mode = optarg[0];
    }
    else {
      fprintf(stderr, "Usage: trial [-d var] \
[-c NUMBER_PASSES]\n");
      fprintf(stderr, "-d X/I/K/D/S tests only xy/in_d/kernel/\
depth/stride to greater depth\n");
      return 2;
    }
  }

  int xy = 16;
  int in_d = 3;
  int k = 5;
  int d = 16;
  int s = 2;
  FILE *out = fopen("timing_data.csv", "w");

  switch (mode) {
    case 'X':
      for(xy=1; xy<=32; xy++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,xy);
      break;
    case 'I':
      for(in_d=1; in_d<=32; in_d++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,in_d);
      break;
    case 'K':
      for(k=1; k<=32; k+=2)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,k);
      break;
    case 'D':
      for(d=1; d<=32; d++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,d);
      break;
    case 'S':
      k=16;
      for(s=1; s<=k; s++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,s);
      break;
    case 'A':
      d = 8;
      s = 1;
      for(xy=1; xy<=16; xy++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,xy);
      fprintf(out,"\n");
      xy = 8;
      for(in_d=1; in_d<=16; in_d++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,in_d);
      fprintf(out,"\n");
      in_d = 3;
      for(k=1; k<=5; k++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,k);
      fprintf(out,"\n");
      k = 5;
      for(d=1; d<=16; d++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,d);
      fprintf(out,"\n");
      d = 8;
      for(s=1; s<=k; s++)
        time_net(xy,in_d,k,d,s,NUM_PASSES,out,s);
      break;
    default:
      fprintf(stderr,"invalid var");
      fprintf(stderr, "Usage: trial [-d var] \
[-c NUMBER_PASSES]\n");
      fprintf(stderr, "-d X/I/K/D/S tests only xy/in_d/kernel/\
depth/stride to greater depth\n");
  }
}
