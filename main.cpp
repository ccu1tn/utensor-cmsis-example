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

void conv(q7_t *Im_in, uint16_t dim_im_in_x, uint16_t dim_im_in_y,
           q7_t *wt, uint16_t dim_kernel_x, uint16_t dim_kernel_y,
           q15_t *Im_out, uint16_t dim_im_out_x, uint16_t dim_im_out_y,
           bool pad_same = true)
{
  uint16_t ch_im_in = 1;                     //number of input tensor channels
  uint16_t ch_im_out = 1;             //number of filters, i.e., output tensor channels
  uint16_t padding_x = 0;             //padding size x
  uint16_t padding_y = 0;             //padding size y
  uint16_t stride_x = 1;              //convolution stride x
  uint16_t stride_y = 1;              //convolution stride y
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
//im = magic(5) * 10 - 128
//w = [10 -10; -7 7]
//conv2(im, w, 'same')
void test_conv_magic5() {
  q7_t Im_in[25] = {
                     42,   112,  -118,   -48,    22,
                    102,   -78,   -58,    12,    32,
                    -88,   -68,     2,    72,    92,
                    -28,    -8,    62,    82,   -98,
                    -18,    52,   122,  -108,   -38
                    };   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {7, -7, -10, 10};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}

  //  712  -568  -848    72   216
  // -528  -808   112   832   496
  // -768   -48   872   592   -24
  //   -8   912   632  -648  -544
  //  136   696    56  -584  -152
void test_conv_magic5_2() {
  q7_t Im_in[25] = {
                     42,   112,  -118,   -48,    22,
                    102,   -78,   -58,    12,    32,
                    -88,   -68,     2,    72,    92,
                    -28,    -8,    62,    82,   -98,
                    -18,    52,   122,  -108,   -38
                    };   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {4, 4, 4, 4};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}

  // 1024   -336   -896    -56    216
  // -256   -816   -176    464    496
  // -736   -296    544    984    -24
  // -216    424   1064   -496   -544
  //  136    696     56   -584   -152
  //w = [0 4; 4 4]
void test_conv_magic5_3() {
  q7_t Im_in[25] = {
                     42,   112,  -118,   -48,    22,
                    102,   -78,   -58,    12,    32,
                    -88,   -68,     2,    72,    92,
                    -28,    -8,    62,    82,   -98,
                    -18,    52,   122,  -108,   -38
                    };   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {0, 4, 4, 4};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}

  //  630  -120  -720   -70   152
  //  -70  -620  -120   380   312
  // -620  -220   430   630   172
  // -120   380   630  -220  -468
  //   84   574   164  -546  -152
//w = [1 2; 3 4]
void test_conv_magic5_4() {
  q7_t Im_in[25] = {
                     42,   112,  -118,   -48,    22,
                    102,   -78,   -58,    12,    32,
                    -88,   -68,     2,    72,    92,
                    -28,    -8,    62,    82,   -98,
                    -18,    52,   122,  -108,   -38
                    };   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {4, 3, 2, 1};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}

//im = ones(5) * 256 - 128 - 1
//w = [1 1; 1 1]
//conv2(im, w, 'same')
void test_conv_ones5_box1() {
  q7_t Im_in[25] = {127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127};   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {1, 1, 1, 1};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}

//im = ones(5) * 256 - 128 - 1
//w = [4 4; 4 4]
//conv2(im, w, 'same')
void test_conv_ones5_box2() {
  q7_t Im_in[25] = {127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127};   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {4, 4, 4, 4};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}

  //  1270   1270   1270   1270    254
  //  1270   1270   1270   1270    254
  //  1270   1270   1270   1270    254
  //  1270   1270   1270   1270    254
  //   635    635    635    635    127
void test_conv_ones5_box3() {
  q7_t Im_in[25] = {127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127,
                    127,   127,   127,   127,   127};   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {4, 4, 1, 1};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}
void test_conv_alt5_box1() {
  q7_t Im_in[25] = {127,   1,   127,   1,   127,
                    127,   1,   127,   1,   127,
                    127,   1,   127,   1,   127,
                    127,   1,   127,   1,   127,
                    127,   1,   127,   1,   127};   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {4, 4, 4, 4};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}


  //  1008   1008   1008   1008   1016
  //  1008   1008   1008   1008   1016
  //  1008   1008   1008   1008   1016
  //  1008   1008   1008   1008   1016
  //   504    504    504    504    508

void test_conv_alt5_box2() {
  q7_t Im_in[25] = {127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127};   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {4, 4, 4, 4};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}


  //    0     0     0     0     0
  //    0     0     0     0     0
  //    0     0     0     0     0
  //    0     0     0     0     0
  //  512  -512   512  -512   508
void test_conv_alt5_diff1() {
  q7_t Im_in[25] = {127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127};   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {4, -4, -4, 4};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}

  //  756   756   756   756   126
  //  756   756   756   756   126
  //  756   756   756   756   126
  //  756   756   756   756   126
  //  762   762   762   762   127
void test_conv_alt5_diff2() {
  q7_t Im_in[25] = {127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127,
                    127,   -1,   127,   -1,   127};   //im = magic(5)' * 10
                                                        //pointer to input tensor
  uint16_t dim_im_in_x = 5;                  //input tensor dimention x
  uint16_t dim_im_in_y = 5;                  //input tensor dimention y

  q7_t wt[4] = {5, 5, 1, 1};                      //pointer to kernel weights
  
  uint16_t dim_kernel_x = 2;          //filter kernel size x
  uint16_t dim_kernel_y = 2;          //filter kernel size y


  q15_t Im_out[25];                        //pointer to output tensor
  uint16_t dim_im_out_x;          //output tensor dimension x
  uint16_t dim_im_out_y;          //output tensor dimension y

  conv(Im_in, dim_im_in_x, dim_im_in_y,
       wt, dim_kernel_x, dim_kernel_y,
      Im_out, dim_im_out_x, dim_im_out_y,
      true);
}


int main()
{
  printf("Tests Start:");
  test_fc();

  printf("\r\arm_convolve_HWC_q7_basic_nonsquare:\r\n");
  printf("magic5:\r\n");
  test_conv_magic5(); //same
  // printf("test_conv_magic5_2\r\n");
  // test_conv_magic5_2(); //same
  // printf("test_conv_magic5_3\r\n");
  // test_conv_magic5_3(); //same
  // printf("test_conv_magic5_4\r\n");
  // test_conv_magic5_4(); //same
  // printf("ones5 box1\r\n");
  // test_conv_ones5_box1(); //same
  // printf("ones5 box2\r\n");
  // test_conv_ones5_box2(); //same
  // printf("ones5 box3\r\n");
  // test_conv_ones5_box3(); //same
  // printf("test_conv_alt5_box1\r\n");
  // test_conv_alt5_box1(); //same
  // printf("test_conv_alt5_box2\r\n");
  // test_conv_alt5_box2(); //same
  // printf("test_conv_alt5_diff1\r\n");
  // test_conv_alt5_diff1();
  // printf("test_conv_alt5_diff2\r\n");
  // test_conv_alt5_diff2();
  // printf("Tests End\r\n");


  // arm_convolve_HWC_q7_basic_nonsquare_no_shift(Im_in, dim_im_in_x, dim_im_in_y, ch_im_in,
  //                                              wt, ch_im_out, dim_kernel_x, dim_kernel_y,
  //                                              padding_x, padding_y, stride_x, stride_y, conv_bias,
  //                                              Im_out, dim_im_out_x, dim_im_out_y,
  //                                              bufferA, bufferB);
  return 0;
}
