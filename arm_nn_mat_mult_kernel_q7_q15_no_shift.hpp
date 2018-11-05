#include "arm_math.h"
#include "arm_nnfunctions.h"

  /**
   * @brief Matrix-multiplication function for convolution
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always conssists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function does the matrix multiplication with weight matrix
   * and 2 columns from im2col. 
   */

template <class T_OUT=q15_t>
T_OUT     *arm_nn_mat_mult_kernel_q7_q15_no_shift_no_bias(const q7_t * pA,
                                        const q15_t * pInBuffer,
                                        const uint16_t ch_im_out,
                                        const uint16_t numCol_A,
                                        T_OUT * pOut)
{
#define ARM_MATH_DSP
#if defined (ARM_MATH_DSP)
    /* set up the second output pointers */
    T_OUT     *pOut2 = pOut + ch_im_out;

    uint16_t  rowCnt = ch_im_out >> 1;
    /* this loop over rows in A */
    while (rowCnt)
    {
        /* setup pointers for B */
        const q15_t *pB = pInBuffer;
        const q15_t *pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const q7_t *pA2 = pA + numCol_A;

        /* init the sum with bias */
        q31_t     sum = 0;
        q31_t     sum2 = 0;
        q31_t     sum3 = 0;
        q31_t     sum4 = 0;

        uint16_t  colCnt = numCol_A >> 2;
        /* accumulate over the vector */
        while (colCnt)
        {
            q31_t     inA11, inA12, inA21, inA22;
            q31_t     inB1 = *__SIMD32(pB)++;
            q31_t     inB2 = *__SIMD32(pB2)++;

            pA = (q7_t *) read_and_pad((void *)pA, &inA11, &inA12);
            pA2 = (q7_t *) read_and_pad((void *)pA2, &inA21, &inA22);

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);
            sum3 = __SMLAD(inA21, inB1, sum3);
            sum4 = __SMLAD(inA21, inB2, sum4);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);
            sum3 = __SMLAD(inA22, inB1, sum3);
            sum4 = __SMLAD(inA22, inB2, sum4);

            colCnt--;
        }                       /* while over colCnt */
        colCnt = numCol_A & 0x3;
        while (colCnt)
        {
            q7_t      inA1 = *pA++;
            q15_t     inB1 = *pB++;
            q7_t      inA2 = *pA2++;
            q15_t     inB2 = *pB2++;

            sum += inA1 * inB1;
            sum2 += inA1 * inB2;
            sum3 += inA2 * inB1;
            sum4 += inA2 * inB2;
            colCnt--;
        }                       /* while over colCnt */
        *pOut++ = (T_OUT) sum;
        *pOut++ = (T_OUT) sum3;
        *pOut2++ = (T_OUT) sum2;
        *pOut2++ = (T_OUT) sum4;

        /* skip the row computed with A2 */
        pA += numCol_A;
        rowCnt--;
    }                           /* for over ch_im_out */

    /* compute left-over row if any */
    if (ch_im_out & 0x1)
    {
        /* setup pointers for B */
        const q15_t *pB = pInBuffer;
        const q15_t *pB2 = pB + numCol_A;

        /* load the bias */
        q31_t     sum = 0;
        q31_t     sum2 = 0;

        uint16_t  colCnt = numCol_A >> 2;
        while (colCnt)
        {
            q31_t     inA11, inA12;
            q31_t     inB1 = *__SIMD32(pB)++;
            q31_t     inB2 = *__SIMD32(pB2)++;

            pA = (q7_t *) read_and_pad((void *)pA, &inA11, &inA12);

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;
            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);

            colCnt--;
        }
        colCnt = numCol_A & 0x3;
        while (colCnt)
        {
            q7_t      inA1 = *pA++;
            q15_t     inB1 = *pB++;
            q15_t     inB2 = *pB2++;

            sum += inA1 * inB1;
            sum2 += inA1 * inB2;
            colCnt--;
        }

        *pOut++ = (T_OUT) sum;
        *pOut2++ = (T_OUT) sum2;
    }

    pOut += ch_im_out;

    /* return the new output pointer with offset */
    return pOut;
#else
    /* To be completed */
    return NULL;
#endif                          /* ARM_MATH_DSP */

}
