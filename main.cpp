#include "mbed.h"
#include "fc_accm32.hpp"

Timer T;
Serial pc(USBTX, USBRX, 115200);

int main()
{
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

  return 0;
}
