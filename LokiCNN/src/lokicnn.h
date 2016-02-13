/*! \file lokicnn.h
    \brief main header
*/

#ifndef LOKICNN_H
#define LOKICNN_H

#include <stdio.h>
#include <assert.h>
#include "string.h"
#include <stdlib.h>
#include "setting.h"
#include "util.h"

#ifdef LOKI
  #include "max_pooling_layer.h"
  #include "fully_connected_layer_MultiCore.h"
#else
  #include "max_pooling_layer.h"
  #include "fully_connected_layer.h"
#endif

#ifdef GEMM
	#include "convolutional_layer.h"
#else
	#include "convolutional_layer_direct.h"
#endif

#include "network.h"
#include "soft_max_layer.h"
#include "ReLU_layer.h"
#include "average_pooling_layer.h"
#include "lrn_layer.h"
#endif
