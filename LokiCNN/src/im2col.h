/*! \file im2col.h
    \brief construction of matrices from image
*/
#include <stdio.h>
#include "setting.h"

///this is the version of im2col which deals with zero_padding
///in Loki, the compiler cannot optimize away the unnecessary if block checking
///in general purpose CPU, with-or-without this explicit function does not observe a performance difference
void im2col_cpu_zero_padding(const Dtype* data_im,
     unsigned int channels,  unsigned int height,  unsigned int width,
     unsigned ksize,  unsigned int stride, Dtype* data_col) 
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;

    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        int col_index = c * height_col * width_col;
        for (h = 0; h < height_col; ++h) {
            int im_row = h_offset + h * stride;
            // int im_col = w_offset;
            int _offset = width*(im_row + height*c_im) + w_offset; 
            for (w = 0; w < width_col; ++w) {
                data_col[col_index++] = data_im[_offset];
                _offset += stride;
            }
        }
    }
}

///This functions assigns pixel value to the specified matrix position
inline Dtype im2col_get_pixel(Dtype *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

/// \brief construction of matrix from input image data_im
///
///This is taken from Caffe's code
///For a given image with specified convolution kernel parameters
///im2col transforms each kernel to a column in a big matrix
///number of columns equal to the total number of output
///length of column equal to the size of convolutionl kernel
void im2col(Dtype* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, Dtype* data_col) 
{
    if(pad == 0){
        im2col_cpu_zero_padding(data_im, channels, height, width,
            ksize, stride, data_col);
    }
    else{
        int c,h,w;
        int height_col = (height - ksize) / stride + 1;
        int width_col = (width - ksize) / stride + 1;
        
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
        
        int channels_col = channels * ksize * ksize;
        for (c = 0; c < channels_col; ++c) {
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
}