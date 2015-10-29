/*! \mainpage Loki CNN -- A pure C Convolutional-Neural-Network Library
 *
 * \section intro_sec Introduction
 *
 * Loki CNN is a light-weight Convolutional-Neural-Netork library.
 *
 * It is written in pure C, with no extra dependencies
 * Although there are options to link with BLAS library for better performance in convolutional layer
 * The library does only forwarding. It is assumed all the weights are pre-trained, for example, by framework like Caffe
 *
 * The library is created for performance benchmarking of a research processor Loki.
 * Fixed-point arithmetic, as well as floating point arithmetic can be used.
 *
 * \section example
 * 
 * \subsection step1 Construct a Network
 *
 * the following gives a quick example of constructing a LeNET-4 model
 *
 * @code
 * Network* net = make_network(8);
 * network_add(net, make_conv_layer(28, 28, 1, 5, 6, 1, 0));
 * network_add(net, make_max_pool_layer(net->layers[0]->out_sx, net->layers[0]->out_sy, net->layers[0]->out_depth, 2, 2));
 * network_add(net, make_conv_layer(net->layers[1]->out_sx, net->layers[1]->out_sy, net->layers[1]->out_depth, 5, 16, 1, 0));
 * ......
 * network_add(net, make_fc_layer(net->layers[5]->out_sx, net->layers[5]->out_sy, net->layers[5]->out_depth, 10));
 * network_add(net, make_softmax_layer(net->layers[6]->out_sx, net->layers[6]->out_sy, net->layers[6]->out_depth));
 * conv_load(net->layers[0], mnist_conv1_params, mnist_conv1_data);
 * ......
 * fc_load(net->layers[6], mnist_ip2_params, mnist_ip2_data);
 * initialize_network(net, 1);
 * @endcode
 *
 * once the model is constructed, and input data loaded, forwarding can be done like this:
 * @code
 * vol_t* input = ......
 * label_t p = net_predict(net, input);
 * @endcode
 *
 * for a complete example, please see ./examples/
 */

/*! \file network.h
    \.
*/


/*! \fn void network_add(Network* net, Layer* layer)
    \brief add the specified layer to the network, this informs the type of layer to the network
    \param net the network
    \param layer the layer
    *
    * For example, one can do:
    * @code
    * Network* net = make_network(11);
    * Layer* l = make_conv_layer(32, 32, 3, 5, 16, 1, 2);
    * network_add(net, l);
    * @endcode
*/
/*! \fn void initialize_network(Network* net, int batch_size)
    \brief allocate buffer space to the layers in the network
    \param net the network
    \param batch_size how many intermediate images to hold within each buffer, default to 1
    *
    * after all the layers has been added, call initialize_network
    * this routine checks the number of layers added matches with declaration
    *
    * the following fives an example of how to initialize an network
    * @code
    * Network* net = make_network(8);
    * network_add(net, make_conv_layer(28, 28, 1, 5, 6, 1, 0));
    * network_add(net, make_max_pool_layer(net->layers[0]->out_sx, net->layers[0]->out_sy, net->layers[0]->out_depth, 2, 2));
    * network_add(net, make_conv_layer(net->layers[1]->out_sx, net->layers[1]->out_sy, net->layers[1]->out_depth, 5, 16, 1, 0));
    * network_add(net, make_max_pool_layer(net->layers[2]->out_sx, net->layers[2]->out_sy, net->layers[2]->out_depth, 2, 2));
    * network_add(net, make_fc_layer(net->layers[3]->out_sx, net->layers[3]->out_sy, net->layers[3]->out_depth, 120));
    * network_add(net, make_relu_layer(net->layers[4]->out_sx, net->layers[4]->out_sy, net->layers[4]->out_depth));
    * network_add(net, make_fc_layer(net->layers[5]->out_sx, net->layers[5]->out_sy, net->layers[5]->out_depth, 10));
    * network_add(net, make_softmax_layer(net->layers[6]->out_sx, net->layers[6]->out_sy, net->layers[6]->out_depth));
    * conv_load(net->layers[0], mnist_conv1_params, mnist_conv1_data);
    * conv_load(net->layers[2], mnist_conv2_params, mnist_conv2_data);
    * fc_load(net->layers[4], mnist_ip1_params, mnist_ip1_data);
    * fc_load(net->layers[6], mnist_ip2_params, mnist_ip2_data);
    * initialize_network(net, 1);
    * @endcode
  
*/




#ifndef NETWORK_H
#define NETWORK_H
#include <assert.h>
#include "layer.h"
#include "data_structure.h"
#include "activation_functions.h"
#ifdef LOKI
    #include <loki/lokilib.h>
#endif
#include "util.h"

typedef vol_t** batch_t;

typedef struct Network{
    
    int num_layers; /**<number of layers in the network */
    Layer **layers; /**<pointing to inference layers */
    int batch_size; /**<number of batches in one inference, default to 1 */
    batch_t* buffer; /**<memory used to hold intermediate data between layers */
    int h;  /**<height of input data to the network */
    int w;  /**<width of input data to the network */
    int c;  /**<channel(or depth) of input data to the network */

    int index;  /**<index, to track which layer the net is current filled upto */
    int e_index; /**<effective index, to track the number of layers which require explicit memory */
} Network;


/*! \fn Network* make_network(int n)
    \brief This allocates memory for pointers to n layers
    \param n number of layers.
*/
Network* make_network(int n)
{
    Network *net = (Network*)malloc(sizeof(Network));
    net->num_layers = n;
    net->layers = (Layer**)malloc(n * sizeof(Layer*));
    net->index = 0;
    net->e_index = 0;
    return net;
}

void network_add(Network* net, Layer* layer){
    int index = net->index;
    assert(index < net->num_layers);
    net->layers[index] = layer;
    /* if the layer type is Activation, define the activation func
     * of the layer above appropriately
     */
    if(layer->type == RELU){
        assert(index != 0);
        net->layers[index-1]->activation = relu_func;
    }
    else{
        net->e_index += 1;
    }
    net->index += 1;
}

/*! \fn void net_summary(Network* net)
    \brief gives a brief summary of the type of layers in the net and the number of key operations taken within each layer
*/
void net_summary(Network* net){
    Layer **layers = net->layers;
    Layer *layer;
    int num_layers = net->num_layers;
    
    unsigned long num_multiplications = 0;
    unsigned long num_comparison = 0;
    unsigned long num_op;
    for(int i = 0; i < num_layers; i++){
        layer = layers[i];
        switch(layer->type){
            case CONVOLUTIONAL:
                num_op = layer->out_sx * layer->out_sy * layer->out_depth * layer->sx * layer->sy * layer->in_depth;
                fprintf(stderr, "layer %d is CONV layer, performs %lu multiplications \n", i, num_op);
                num_multiplications += num_op;
                break;
            case FULLY_CONNECTED:
                num_op = layer->num_inputs * layer->out_depth;
                fprintf(stderr, "layer %d is FC layer, performs %lu multiplications \n", i, num_op);
                num_multiplications += num_op;
                break;
            case POOLING:
                num_op = layer->out_sx * layer->out_sy * layer->out_depth * layer->sx * layer->sy;
                fprintf(stderr, "layer %d is POOL layer, performs %lu comparison \n", i, num_op);
                num_comparison += num_op;
                break;
            case RELU:
                num_op = layer->out_sx * layer->out_sy * layer->out_depth;
                fprintf(stderr, "layer %d is RELU layer, performs %lu comparison \n", i, num_op);
                num_comparison += num_op;
                break;
            case SOFTMAX:
                fprintf(stderr, "layer %d is SMAX layer \n \n", i);
                break;
            default:
                break;
        }
    }
    fprintf(stderr, "on total, requires: \n\t\t\t%lu multiplications \n\t\t\t%lu comparison \n\n", num_multiplications, num_comparison);    
}

void initialize_network(Network* net, int batch_size){
    // Check number of layers is correct
    int num_layers = net->num_layers;
    assert(net->index == num_layers);
    assert(net->e_index <= net->index);
    assert(batch_size >= 1);

    // Set the height, width, channel parameters
    Layer* top_layer = net->layers[0];
    net->h = top_layer->in_sx;
    net->w = top_layer->in_sy;
    net->c = top_layer->in_depth;
    net->batch_size = batch_size;

    int num_buffer_layers = net->index + 1;
    int e_num_buffer_layers = net->e_index + 1;
    /* allocate space for buffers */
    batch_t* buffer = (batch_t*)malloc(sizeof(vol_t**)*e_num_buffer_layers);
    assert(buffer);
    // allocate enough pointer for each buffer according to batch_size
    for(int i = 0; i < e_num_buffer_layers; i++){
        buffer[i] = (vol_t**)malloc(sizeof(vol_t*)*batch_size);
        assert(buffer[i]);
    }
    // allocate space for top-level data buffer
    fprintf(stderr, "initializing the data layer buffer \n");
    for(int j = 0; j < batch_size; j++){
        buffer[0][j] = make_vol(net->h, net->w, net->c, 0);
    }
    fprintf(stderr, "intializing intermediate layer buffers, total of: %d \n", num_buffer_layers);
    Layer* layer;
    for(int i = 1, e_i = 1; i < num_buffer_layers; i++){
        
        layer = net->layers[i-1];
        // Activation layer can be done in place, no need for extra buffer
        if(layer->type == RELU){
            debug_print("%d layer is activation layer, in place activation to reduce memory requirement \n", i);
            continue;
        }
        int sx = layer->out_sx;
        int sy = layer->out_sy;
        int depth = layer->out_depth;

        for(int j = 0; j < batch_size; j++){
            buffer[e_i][j] = make_vol(sx, sy, depth, 0);
        }
        e_i++;
        debug_print("%d layer buffer initialized, dimension is %d %d %d \n", i, sx, sy, depth);
    }
    net->buffer = buffer;
    fprintf(stderr, "network initialized successfully \n");
}

/*! \fn void free_network(Network* net)
    \brief free the allocated buffer space 
*/
void free_network(Network* net) {
    // free the memory allocated for each layer
    for(int i = 0; i < net->num_layers; i++){        
        free(net->layers[i]);
    }
    // free the memory allocated for buffers
    batch_t* buffer = net->buffer;
    for(int j = 0; j < net->e_index; j++){
        for(int z = 0; z < net->batch_size; z++){
            free_vol(buffer[j][z]);
        }
        free(buffer[j]);
    }
    free(buffer);
    free(net);
}

/*! \fn void net_forward(Network* net)
    \brief forward from the data layer
*/
void net_forward(Network* net) {
    Layer **layers = net->layers;
    Layer *layer;
    int num_layers = net->index;
    for(int i = 0, e_i = 0 ; i < num_layers; i++){
        layer = layers[i];
        if(layer->type == RELU){
            continue;
        }
        debug_print("forwarding from layer %d \n", i);
        #ifdef LOKI
            unsigned long cycle_count = get_cycle_count();
            (*layer->forward)(layer, net->buffer[e_i], net->buffer[e_i+1], 0, 0);
            debug_print("takes %lu cycle to complete layer %d \n", get_cycle_count()-cycle_count, i);
        #else
            generic_forward_func(layer, net->buffer[e_i], net->buffer[e_i+1], 0, 0);
        #endif
        e_i ++;
    }
}

/*! \fn int net_num_category(Network* net)
    \brief return the number of categories the network is categorizing
*/
inline int net_num_category(Network* net){
  int num = net->layers[net->num_layers-1]->out_depth;
  return num;
}

/*! \fn void net_predict(Network* net)
    \brief forward and make prediction based on the last layer
    \brief this assumes the last layer is a softmax layer
*/
label_t net_predict(Network* net){
    net_forward(net);
    batch_t prediction_vector = net->buffer[net->e_index];
    int num_category = net_num_category(net);
    label_t prediction = 0;
    storage_t max_prob = 0;
    for(int i = 0; i < num_category; i++){
        storage_t prob = get_vol(prediction_vector[0], 0, 0, i);
        if(prob > max_prob){
            prediction = i;
            max_prob = prob;
        }
    }
    return prediction;
}

/*! \fn void net_test(Network* net, vol_t** input, label_t* labels, int n)
    \brief forward the given inputs and output the accuracy analysis
    \param net the network
    \param input the input images
    \param label the actual input classification
    \param n the size of input
*/
void net_test(Network* net, vol_t** input, label_t* labels, int n) {
    fprintf(stderr, "Testing the accuracy of %d inferences...\n", n);
    
    uint64_t start_time = timestamp_us(); 

    int num_correct = 0;
    for (int i = 0; i < n; i++) {
        copy_vol(net->buffer[0][0], input[i]);    // everytime, set the input at data_layer
        label_t predicted = net_predict(net);      // run_prediction
        label_t actual = labels[i];
        
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
