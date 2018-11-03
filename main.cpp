#include "mbed.h"
#include "fc_accm32.hpp"
#include "conv_accm32.hpp"
#include "arm_nnfunctions.h"
#include "stdlib.h"
#include <math.h>

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

void test_conv(void) {
  q7_t Im_in[16] = {16,    5,    9,    4,
                    2,   11,    7,   14,
                    3,   10,    6,   15,
                  13,    8,   12,    1};   //1 //pointer to input tensor
  uint16_t dim_im_in_x = 4;                  //input tensor dimention x
  uint16_t dim_im_in_y = 4;                  //input tensor dimention y
  uint16_t ch_im_in = 1;                     //number of input tensor channels
  q7_t wt[4] = {1, -1, -1, 1};                      //pointer to kernel weights
  uint16_t ch_im_out = 1;             //number of filters, i.e., output tensor channels
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y
  uint16_t padding_x = 0;             //padding size x
  uint16_t padding_y = 0;             //padding size y
  uint16_t stride_x = 1;              //convolution stride x
  uint16_t stride_y = 1;              //convolution stride y
  q7_t conv_bias[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};                    //pointer to bias
  q7_t Im_out[16];                        //pointer to output tensor
  uint16_t dim_im_out_x = 4;          //output tensor dimension x
  uint16_t dim_im_out_y = 4;          //output tensor dimension y
  q15_t bufferA[16];                   // 2 * 2 * 2?      //pointer to buffer space for input
  q7_t  bufferB[16];                 //pointer to buffer space for output

  uint16_t bias_shift = 0;
  uint16_t out_shift = 0;

  arm_convolve_HWC_q7_basic_nonsquare(Im_in, dim_im_in_x, dim_im_in_y, ch_im_in,
                                      wt, ch_im_out, dim_kernel_x, dim_kernel_y,
                                      padding_x, padding_y, stride_x, stride_y, conv_bias,
                                      bias_shift, out_shift,
                                      Im_out, dim_im_out_x, dim_im_out_y,
                                      bufferA, bufferB);

  printf("\r\arm_convolve_HWC_q7_basic_nonsquare:\r\n");
  printf("out result:\r\n");
  for(uint32_t y = 0; y < dim_im_out_y; y++) {
    for(uint32_t x = 0; x < dim_im_out_x; x++) {
      auto tmp = Im_out[x + y * dim_im_out_y];

      uint8_t pad = 4; //alignment with max padding of 4
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
  test_conv();
  printf("Tests End");


  // arm_convolve_HWC_q7_basic_nonsquare_no_shift(Im_in, dim_im_in_x, dim_im_in_y, ch_im_in,
  //                                              wt, ch_im_out, dim_kernel_x, dim_kernel_y,
  //                                              padding_x, padding_y, stride_x, stride_y, conv_bias,
  //                                              Im_out, dim_im_out_x, dim_im_out_y,
  //                                              bufferA, bufferB);
  return 0;
}
