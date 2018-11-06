#include "mbed.h"
#include "fc_accm32.hpp"
#include "conv_accm32.hpp"
#include "arm_nnfunctions.h"
#include "stdlib.h"
#include <math.h>
#include <algorithm>

Timer T;
Serial pc(USBTX, USBRX, 115200);

void test_fc(void) {
  q7_t pV[4] = {10, 20, 30, 40};
  q7_t pM[16] = {16,    5,    9,    4,
                  2,   11,    7,   14,
                  3,   10,    6,   15,
                13,    8,   12,    1}; //magic(4)'

  uint16_t dim_vec = 4;
  uint16_t num_of_rows = 4;
  q7_t bias[4] = {-128 , 127, -1, -1};
  uint32_t pOut[4]; //input * magic(4) + bias = 562   1137   1009    689
  q15_t vec_buffer[4];

  arm_fully_connected_no_shift(pV, pM, dim_vec, num_of_rows, bias, pOut, vec_buffer);

  printf("\r\narm_fully_connected_q7_test:\r\n");
  printf("out result: %d %d, %d, %d\r\n", pOut[0], pOut[1], pOut[2], pOut[3]);
}

void test_conv(bool pad_same = true) {
  q7_t Im_in[25] = { 42,   102,   -88,   -28,   -18,
                    112,   -78,   -68,    -8,    52,
                   -118,   -58,     2,    62,   122,
                    -48,    12,    72,    82,  -108,
                     22,    32,    92,   -98,   -38};   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y
  uint16_t ch_im_in = 1;                     //number of input tensor channels
  q7_t wt[4] = {10, -10, -7, 7};                      //pointer to kernel weights
  uint16_t ch_im_out = 1;             //number of filters, i.e., output tensor channels
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y
  uint16_t padding_x = 0;             //padding size x
  uint16_t padding_y = 0;             //padding size y
  uint16_t stride_x = 1;              //convolution stride x
  uint16_t stride_y = 1;              //convolution stride y
  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y
  q15_t bufferA[64];                   // 2 * 2 * 2?      //pointer to buffer space for input //???
  q7_t  bufferB[64];                 //pointer to buffer space for output  //???
  uint16_t pad_along_height;
  uint16_t pad_along_width;

  if (pad_same) {
    dim_im_out_y = ceil(float(dim_im_in_y) / float(stride_y));
    dim_im_out_x  = ceil(float(dim_im_in_x) / float(stride_x));

    if (dim_im_out_y % stride_y == 0) {
      pad_along_height = max(dim_kernel_y - stride_y, 0);
    } else {
      pad_along_height = max(dim_kernel_y - (dim_im_in_y % stride_y), 0);
    }
    
    if (dim_im_out_x % stride_x == 0) {
      pad_along_width = max(dim_kernel_x - stride_x, 0);
    } else {
      pad_along_width = max(dim_kernel_x - (dim_im_in_x % stride_x), 0);
    }

    padding_x = pad_along_width / 2;
    padding_y = pad_along_height / 2;
  } else {
    dim_im_out_y = ceil(float(dim_im_in_y - dim_kernel_y + 1) / float(stride_y));
    dim_im_out_x = ceil(float(dim_im_in_x - dim_kernel_x + 1) / float(stride_x));
  }


  arm_convolve_HWC_q7_basic_nonsquare_no_shift_no_bias(Im_in, dim_im_in_x, dim_im_in_y, ch_im_in,
                                    wt, ch_im_out, dim_kernel_x, dim_kernel_y,
                                    padding_x, padding_y, stride_x, stride_y,
                                    Im_out, dim_im_out_x, dim_im_out_y,
                                    bufferA, bufferB);
  

  for(uint32_t y = 0; y < dim_im_out_y; y++) {
    for(uint32_t x = 0; x < dim_im_out_x; x++) {
      auto tmp = Im_out[x + y * dim_im_out_y];

      uint8_t pad = 6; //alignment with max padding of 4
      if(tmp != 0) pad = pad - log10(abs(tmp));
      if(tmp <= 0) pad -= 1;
      for(; pad > 0; pad=pad-1) 
        printf(" ");
      //end of alignment code
      
      printf("%d ", tmp);
    }
    printf("\r\n");
  }
}

int main()
{
  printf("Tests Start:");
  test_fc();

  printf("\r\arm_convolve_HWC_q7_basic_nonsquare:\r\n");
  printf("same:\r\n");
  test_conv(true); //same
  printf("valid:\r\n");
  test_conv(false); //same

  printf("Tests End\r\n");


  // arm_convolve_HWC_q7_basic_nonsquare_no_shift(Im_in, dim_im_in_x, dim_im_in_y, ch_im_in,
  //                                              wt, ch_im_out, dim_kernel_x, dim_kernel_y,
  //                                              padding_x, padding_y, stride_x, stride_y, conv_bias,
  //                                              Im_out, dim_im_out_x, dim_im_out_y,
  //                                              bufferA, bufferB);
  return 0;
}
