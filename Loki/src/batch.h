/*
 * We organize data as "batches" of volumes. Each batch consists of a number of samples,
 * each of which contains a volume for every intermediate layer. Say we have L layers
 * and a set of N input images. Then batch[l][n] contains the volume at layer l for
 * input image n.
 *
 * By using batches, we can process multiple images at once in each run of the forward
 * functions of the different layers.
 */
#include "data_structure.h"