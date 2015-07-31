// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "data_structure.h"
#include "util.h"

typedef vol_t** batch_t;

typedef struct Network{
    int num_layers;
    Layer **layers;

    int batch_size; // think at the moment defaut to zero will be the best and easiest
    batch_t* buffer;
    int h, w, c;

    //index, to count which layer the net is current filled upto
    int index;
} Network;

Network* make_network(int n)
{
    Network *net = (Network*)malloc(sizeof(Network));
    net->num_layers = n;
    net->layers = (Layer**)malloc(n * sizeof(Layer*));
    net->index = 0;
    return net;
}

void network_add(Network* net, Layer* layer){
    int index = net->index;
    assert(index < net->num_layers);
    fprintf(stderr, "adding the %d layer \n", index);
    net->layers[index] = layer;
    index += 1;
    net->index = index;
}

void initialize_network(Network* net, int batch_size){
    // Check number of layers is correct
    int num_layers = net->num_layers;
    assert(net->index == num_layers);
    assert(batch_size >= 1);
    // TODO: check the input and output of each consecutive layers match

    // Set the height, width, channel parameters
    Layer* top_layer = net->layers[0];
    net->h = top_layer->in_sx;
    net->w = top_layer->in_sy;
    net->c = top_layer->in_depth;
    net->batch_size = batch_size;

    int num_buffer_layers = num_layers + 1;
    
    batch_t* buffer = (batch_t*)malloc(sizeof(vol_t**)*num_buffer_layers);
    
    for(int i = 0; i < num_buffer_layers; i++){
        buffer[i] = (vol_t**)malloc(sizeof(vol_t*)*batch_size);
    }
    fprintf(stderr, "initialize the data layer buffer \n");
    for(int j = 0; j < batch_size; j++){
        buffer[0][j] = make_vol(net->h, net->w, net->c, 0.0);
    }
    fprintf(stderr, "intializing intermediate layer buffers, total of: %d \n", num_buffer_layers);
    Layer* layer;
    for(int i = 1; i < num_buffer_layers; i++){
        layer = net->layers[i-1];

        int sx = layer->out_sx;
        int sy = layer->out_sy;
        int depth = layer->out_depth;
        for(int j = 0; j < batch_size; j++){
            buffer[i][j] = make_vol(sx, sy, depth, 0.0);
        }
        fprintf(stderr, "%d layer buffer initialized, dimension is %d %d %d \n", i, sx, sy, depth);
    }
    net->buffer = buffer;

    fprintf(stderr, "network initialized successfully \n");
}

/* 
 * Free our specific CNN.
 */

void free_network(Network* net) {
    // free the memory allocated for each layer
    for(int i = 0; i < net->num_layers; i++){        
        free(net->layers[i]);
    }
    // free the memory allocated for buffers
    batch_t* buffer = net->buffer;
    for(int j = 0; j < net->num_layers + 1; j++){
        for(int z = 0; z < net->batch_size; z++){
            free_vol(buffer[j][z]);
        }
        free(buffer[j]);
    }
    free(buffer);

    free(net);
}

void net_forward(Network* net) {
    Layer **layers = net->layers;
    Layer *layer;
    int num_layers = net->num_layers;
    for(int i = 0; i < num_layers; i++){
        layer = layers[i];
        (*layer->forward)(layer, net->buffer[i], net->buffer[i+1], 0, net->batch_size - 1);
    }
}

int net_num_category(Network* net){
  int num = net->layers[net->num_layers-1]->out_depth;
  return num;
}

label_t net_predict(Network* net){
    net_forward(net);
    batch_t prediction_vector = net->buffer[net->num_layers];
    int num_category = net_num_category(net);
    
    label_t prediction = 0;
    double max_prob = 0.0;
    for(int i = 0; i < num_category; i++){
        double prob = get_vol(prediction_vector[0], 0, 0, i);
        // fprintf(stderr, "the prob for category %d is %f \n", i, prob);
        if(prob > max_prob){
            prediction = i;
            max_prob = prob;
        }
    }
    return prediction;
}

void net_predict_Multiple(Network* net, vol_t** input, int n){
    fprintf(stderr, "Testing the speed of %d inferences...\n", n);
    
    uint64_t start_time = timestamp_us(); 
    for (int i = 0; i < n; i++) {
        copy_vol(net->buffer[0][0], input[i]);    // everytime, set the input at data_layer
        net_predict(net);
    }
    uint64_t end_time = timestamp_us();

    double dt = (double)(end_time-start_time) / 1000.0;
    fprintf(stderr, "\nTIME: %.2lf ms\n", dt);
    fprintf(stderr, "\nTime/Image %.2lf ms \n\n", (dt/ (double)n));   
}

void net_test(Network* net, vol_t** input, label_t* labels, int n) {
    fprintf(stderr, "Testing the accuracy of %d inferences...\n", n);
    
    uint64_t start_time = timestamp_us(); 

    int num_correct = 0;
    for (int i = 0; i < n; i++) {
        copy_vol(net->buffer[0][0], input[i]);    // everytime, set the input at data_layer
        label_t predicted = net_predict(net);      // run_prediction
        label_t actual = labels[i];
#ifdef DEBUG
  printf("%d,%d\n",predicted,actual);
#endif
        if(predicted == actual){
            num_correct +=1;
        }
    }

    uint64_t end_time = timestamp_us();

    fprintf(stderr, "%d of correct prediction out of %d trials \n", num_correct, n);
    double dt = (double)(end_time-start_time) / 1000.0;
    fprintf(stderr, "\nTIME: %.2lf ms\n", dt);
    fprintf(stderr, "\nTime/Image %.2lf ms \n\n", (dt/ (double)n));
}
#endif
