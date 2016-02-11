/*! \file setting.h
    \brief configuration file for datatype, algorithms, etc.
*/

/*! \def FIX16
    \brief options are FIX8, FIX16, FLOAT, DOUBLE
    \brief if loki is set with parallel option, must be FIX8 or FIX16
*/
/*! \def DEBUG
    \brief flag for printing debug information e.g. number of cycles taken by each layer, will be printed to stderr
*/
/*! \var typedef double weight_t
    \brief the type of input weights, user specified
*/
/*! \var typedef storage_t Dtype
	\brief the type used in Matrix multiplication
*/
/*! \var typedef int32_t storage_t
	\brief int32_t if FIXED_POINT, float if FLOAT, double if DOUBLE
	\brief the type used in Matrix multiplication
*/
/*! \var const int CONV_LAYER_NUM_CORE
	\brief maximum number of cores used in convolution layer
*/
/*! \var const int FC_LAYER_NUM_CORE
	\brief maximum number of cores used in fully connected layer
*/
/*! \var const int MAX_POOLING_LAYER_NUM_CORE
	\brief maximum number of cores used in pooling layer
*/
#ifndef SETTING_H
#define SETTING_H
/*
 * choose the data representation, either DOUBLE, FLOAT, OR FIX16
 */
#include <stdint.h>
#include <math.h>
/*!
    \brief flag for selecting datatype, currently support FLOAT, DOUBLE, FIX16, FIX8
*/
#define FIX8

/*!
    \brief flag for printing debug information e.g. number of cycles taken by each layer, will be printed to stderr
*/
#define DEBUG 0

/*! 
    \brief flag for choosing which algorithm to use in convolutional layer, currently support GEMM, DIRECT
*/
#define GEMM

#if defined(DOUBLE)
  	typedef double storage_t;
#elif defined(FLOAT)
  	typedef float storage_t;
#elif defined(FIX16)
  	typedef int32_t storage_t;
#elif defined(FIX8)
  	typedef int32_t storage_t;
#endif

/*!
    \brief flag for printing debug information e.g. number of cycles taken by each layer, will be printed to stderr
*/
typedef double weight_t;

typedef storage_t Dtype;

const int FC_LAYER_NUM_CORE = 1;
const int MAX_POOLING_LAYER_NUM_CORE = 1;
const int IM2COL_NUM_CORE = 1;
const int CONV_NUM_CORE = 1;

/*!
    \brief flag determining whether or not to initialize memory for storage of image and weights
*/
const int WEIGHTS_INITIALIZE_TO_ZERO = 0;
#endif