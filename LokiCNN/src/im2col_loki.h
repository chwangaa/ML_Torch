#ifndef IM2COL_H
#define IM2COL_H
#include <stdio.h>

#define IM2COL_NUM_CORE 8

typedef struct im2col_global_data_ {
    Dtype* data_im;
    unsigned int channels;
    unsigned int height;
    unsigned int width;
    unsigned ksize;
    unsigned int stride;
    Dtype* data_col;
    int pad;
    int cores;
} im2col_global_data;


///refer to im2col.h for reference
void im2col_loki_zero_padding(const Dtype* data_im,
     unsigned int channels,  unsigned int height,  unsigned int width,
     unsigned ksize,  unsigned int stride, Dtype* data_col, int cores) 
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    int core = get_core_id();
    /* this is where parallelization takes place */
    for (c = core; c < channels_col; c+=cores) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        int col_index = c * height_col * width_col;
        for (h = 0; h < height_col; ++h) {
            int im_row = h_offset + h * stride;
            int _offset = width*(im_row + height*c_im) + w_offset; 
            // explicit loop unroll in unit of 4, utilizing 4 channels
            // maybe it is better to use 6 channels, again depends on
            // the size of the width itself.
            for (w = 0; w < width_col - 4; w+=4) {
              asm volatile(
                  "fetchr 1f \n"
                  "addu %0, %0, r0 -> 10 \n"
                  "addu %0, %0, %2 -> 11 \n"
                  "addu %0, %0, %2 -> 12 \n"
                  "addu %0, %0, %2 -> 13 \n"

                  "stw r4, 0x0(%1) -> 10\n"
                  "stw r5, 0x4(%1) -> 11\n"
                  "stw r6, 0x8(%1) -> 12\n"
                  "stw.eop r7, 0xc(%1) ->13\n"
                  "1:"
                  : 
                  : "r"(&data_im[_offset]), "r"(&data_col[col_index]), "r"(stride*4)
                );
              _offset += 4*stride;
              col_index += 4;
            }
            
            // Final few iterations.
            for(; w < width_col; w++){
                data_col[col_index++] = data_im[_offset];
                _offset += stride;
            }
        }
    }
}

///refer to im2col.h for reference
inline Dtype im2col_get_pixel(Dtype *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

///refer to im2col.h for reference
void im2col_workerCore(const void* data) 
{
    int core = get_core_id();
    int addr = loki_mem_address(0, core, CH_REGISTER_4, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(10, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_5, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(11, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_6, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(12, addr);
    addr = loki_mem_address(0, core, CH_REGISTER_7, GROUPSIZE_8, false, false, false, false);  
    set_channel_map(13, addr);
  
    im2col_global_data* d = (im2col_global_data*)data;
    if(d->pad == 0){
        im2col_loki_zero_padding(d->data_im, d->channels, d->height, d->width,
            d->ksize, d->stride, d->data_col, d->cores);
    }
    else{
        Dtype* data_im = d->data_im;
        unsigned int channels = d->channels;
        unsigned int height = d->height;
        unsigned int width = d->width;
        unsigned ksize = d->ksize;
        unsigned int stride = d->stride;
        Dtype* data_col = d->data_col;
        unsigned pad = d->pad;
        int cores = d->cores;
        uint core = get_core_id() + 8*tile2int(get_tile_id());
        
        int c,h,w;
        int height_col = (height - ksize) / stride + 1;
        int width_col = (width - ksize) / stride + 1;
        
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
        
        int channels_col = channels * ksize * ksize;
        /* this is where to parallelization takes place */
        for (c = core; c < channels_col; c+=cores) {
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = 0; h < height_col; ++h) {
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                            im_row, im_col, c_im, pad);
                }
            }
        }
    }
    loki_tile_sync(d->cores);
}

///this is Loki's one-tile parallel version of im2col, each core takes a column and computes it
void im2col(const Dtype* data_im,
     unsigned int channels,  unsigned int height,  unsigned int width,
     unsigned ksize,  unsigned int stride, int pad, Dtype* data_col) {  
  
  const int cores = IM2COL_NUM_CORE;
  loki_init_default(cores, 0);

  /* construct a closure for the worker core */
  im2col_global_data* data = malloc(sizeof(im2col_global_data));
  data->data_im = data_im;
  data->channels = channels;
  data->height = height;
  data->width = width;
  data->ksize = ksize;
  data->stride = stride;
  data->pad = pad;
  data->data_col = data_col;
  data->cores = cores;
  
  distributed_func* config = malloc(sizeof(distributed_func));
  config->cores = cores;
  config->func = &im2col_workerCore;
  config->data = data;
  config->data_size = sizeof(im2col_global_data);
  loki_execute(config);
}

#endif
