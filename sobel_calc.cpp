#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
#include <arm_neon.h>
using namespace cv;


// defines

#define BLUE_SCALE 29
#define GREEN_SCALE 150
#define RED_SCALE 77
/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out)
{
  double color;

  // #define IMG_WIDTH 640
  // #define IMG_HEIGHT 480
  // gray = 0.299*R + 0.587*G + 0.114*B

  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<img.cols; j++) {
      color = .114*img.data[STEP0*i + STEP1*j] +
              .587*img.data[STEP0*i + STEP1*j + 1] +
              .299*img.data[STEP0*i + STEP1*j + 2];
      img_gray_out.data[IMG_WIDTH*i + j] = color;
    }
  }

  // have to split into lower and upper then combine?
  // each pixel is 8 bits? (16 * 8 = 128)
  // when we mulitply we should keep the product in a double width reg

  // // load the bgr values as integers to avoid using floats. floats are bad

  // uint8x8_t blue_w = vdup_n_u8(BLUE_SCALE);
  // uint8x8_t green_w = vdup_n_u8(GREEN_SCALE);
  // uint8x8_t red_w = vdup_n_u8(RED_SCALE);

  // // this comes out to 307,200. divided by 16 is 0. so traversing will not leave us out of bounds
  // // also this mean that each pixel gas a "sub array" that has [BGR]. So the total total is the total_pixel * 3

  // int total_pixel = img.rows * img.cols; 
  
  // uint8_t *s_ptr = img.data;
  // uint8_t *d_ptr = img_gray_out.data;

  // // striding 16 to optimize. 1 pixel is 1 byte of data that contains bgr so 3 bytes of that. 16 * 3 is 48 total bytes.
  // // data is interleved so i need to strip the bgr into there own lanes?
  // // also i need to have this upper and lower logic to account for the multiply?
  // for (int i = 0; i < total_pixel; i += 16) {
  //   // need to keep track of the pixels as we jumo so do we use a pointer?
  //   // the array that we grab has 48 total bytes with BGRBGRBGR... values, so 16 pixels is actaull 48 bytes worht of data
    
  //   // to get 16 pixels it is actuall 48 bytes of data that we will put into 3 vectors
  //   // ptr arithmatic is supposed to traverse every 48 bytes, holding 16 bytes in each reg so it we will have to use the q
  //   uint8x16x3_t bgr_vals = vld3q_u8(s_ptr + (i * 3)); // if i = 0 then it will load first 16 pix

  //   // strat will be similar to 271 where we have a double accum storage for the upper and lower
  //   // multiplying whole will exceed 256 bits? I think this is bc if we had 128 bit then the sum will need to be 256 as padding/buffer so if we split into 2 64 then we can buffer with 128 accum

  //   // first do the lower accum, type is 16 bits and 8 lanes since we are splitting the 16 pixels into 2 8's

  //   uint16x8_t accum_lower_blue = vmull_u8(vget_low_u8(bgr_vals.val[0]), blue_w);
  //   uint16x8_t accum_lower_green = vmull_u8(vget_low_u8(bgr_vals.val[1]), green_w);
  //   uint16x8_t accum_lower_red = vmull_u8(vget_low_u8(bgr_vals.val[2]), red_w);

  //   uint16x8_t accum_gray_low_temp= vaddq_u16(accum_lower_blue, accum_lower_green);
  //   uint16x8_t accum_gray_low = vaddq_u16(accum_gray_low_temp, accum_lower_red);

  //   // now we can do the upper

  //   uint16x8_t accum_upper_blue = vmull_u8(vget_high_u8(bgr_vals.val[0]), blue_w);
  //   uint16x8_t accum_upper_green = vmull_u8(vget_high_u8(bgr_vals.val[1]), green_w);
  //   uint16x8_t accum_upper_red = vmull_u8(vget_high_u8(bgr_vals.val[2]), red_w);

  //   uint16x8_t accum_gray_upper_temp= vaddq_u16(accum_upper_blue, accum_upper_green);
  //   uint16x8_t accum_gray_upper = vaddq_u16(accum_gray_upper_temp, accum_upper_red);

  //   // convert to floats
  //   uint8x8_t result_lower = vshrn_n_u16(accum_gray_low, 8);
  //   uint8x8_t result_upper = vshrn_n_u16(accum_gray_upper, 8);

  //   uint8x16_t result = vcombine_u8(result_lower, result_upper); 
  //   vst1q_u8(d_ptr + i, result);
  // }
}

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/
// void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
// {
//   Mat img_outx = img_gray.clone();
//   Mat img_outy = img_gray.clone();

//   // Apply Sobel filter to black & white image
//   unsigned short sobel;

//   // Calculate the x convolution
//   for (int i=1; i<img_gray.rows; i++) {
//     for (int j=1; j<img_gray.cols; j++) {
//       sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
// 		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
// 		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
// 		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
// 		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
// 		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

//       sobel = (sobel > 255) ? 255 : sobel;
//       img_outx.data[IMG_WIDTH*(i) + (j)] = sobel;
//     }
//   }

//   // Calc the y convolution
//   for (int i=1; i<img_gray.rows; i++) {
//     for (int j=1; j<img_gray.cols; j++) {
//      sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
// 		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
// 		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
// 		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
// 		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
// 		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

//      sobel = (sobel > 255) ? 255 : sobel;

//      img_outy.data[IMG_WIDTH*(i) + j] = sobel;
//     }
//   }

//   // Combine the two convolutions into the output image
//   for (int i=1; i<img_gray.rows; i++) {
//     for (int j=1; j<img_gray.cols; j++) {
//       sobel = img_outx.data[IMG_WIDTH*(i) + j] +
// 	img_outy.data[IMG_WIDTH*(i) + j];
//       sobel = (sobel > 255) ? 255 : sobel;
//       img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
//     }
//   }
// }

void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
{
  // Loop Fusion
  
  
  //NEON Vectorization
  // Process 8 pixels at a time using NEON intrinsics
  
  uint8_t* img_data = img_gray.data;
  uint8_t* out_data = img_sobel_out.data;
  
  // Process the image row by row
  for (int i = 1; i < img_gray.rows - 1; i++) {
    int j = 1;
    
    // Process 8 pixels at a time with NEON
    for (; j <= img_gray.cols - 9; j += 8) {
      // Load the 3x3 neighborhood for 8 pixels
      // Row i-1
      uint8x8_t top_left = vld1_u8(&img_data[IMG_WIDTH*(i-1) + (j-1)]);
      uint8x8_t top_mid = vld1_u8(&img_data[IMG_WIDTH*(i-1) + j]);
      uint8x8_t top_right = vld1_u8(&img_data[IMG_WIDTH*(i-1) + (j+1)]);
      
      // Row i
      uint8x8_t mid_left = vld1_u8(&img_data[IMG_WIDTH*i + (j-1)]);
      uint8x8_t mid_right = vld1_u8(&img_data[IMG_WIDTH*i + (j+1)]);
      
      // Row i+1
      uint8x8_t bot_left = vld1_u8(&img_data[IMG_WIDTH*(i+1) + (j-1)]);
      uint8x8_t bot_mid = vld1_u8(&img_data[IMG_WIDTH*(i+1) + j]);
      uint8x8_t bot_right = vld1_u8(&img_data[IMG_WIDTH*(i+1) + (j+1)]);
      
      // Calculate Gx = [-1 0 1; -2 0 2; -1 0 1]
      // Gx = -top_left + top_right - 2*mid_left + 2*mid_right - bot_left + bot_right
      
      // Promote to 16-bit for calculations
      int16x8_t top_left_s = vreinterpretq_s16_u16(vmovl_u8(top_left));
      int16x8_t top_right_s = vreinterpretq_s16_u16(vmovl_u8(top_right));
      int16x8_t mid_left_s = vreinterpretq_s16_u16(vmovl_u8(mid_left));
      int16x8_t mid_right_s = vreinterpretq_s16_u16(vmovl_u8(mid_right));
      int16x8_t bot_left_s = vreinterpretq_s16_u16(vmovl_u8(bot_left));
      int16x8_t bot_right_s = vreinterpretq_s16_u16(vmovl_u8(bot_right));
      int16x8_t top_mid_s = vreinterpretq_s16_u16(vmovl_u8(top_mid));
      int16x8_t bot_mid_s = vreinterpretq_s16_u16(vmovl_u8(bot_mid));
      
      // Calculate Gx
      int16x8_t gx_s = vsubq_s16(top_right_s, top_left_s);
      int16x8_t mid_diff = vsubq_s16(mid_right_s, mid_left_s);
      mid_diff = vshlq_n_s16(mid_diff, 1); // multiply by 2
      gx_s = vaddq_s16(gx_s, mid_diff);
      int16x8_t bot_diff = vsubq_s16(bot_right_s, bot_left_s);
      gx_s = vaddq_s16(gx_s, bot_diff);
      
      // Take absolute value
      int16x8_t gx_abs = vabsq_s16(gx_s);
      
      // Calculate Gy = [-1 -2 -1; 0 0 0; 1 2 1]
      // Gy = -top_left - 2*top_mid - top_right + bot_left + 2*bot_mid + bot_right
      int16x8_t gy_s = vsubq_s16(bot_left_s, top_left_s);
      int16x8_t vert_diff = vsubq_s16(bot_mid_s, top_mid_s);
      vert_diff = vshlq_n_s16(vert_diff, 1); // multiply by 2
      gy_s = vaddq_s16(gy_s, vert_diff);
      int16x8_t right_diff = vsubq_s16(bot_right_s, top_right_s);
      gy_s = vaddq_s16(gy_s, right_diff);
      
      int16x8_t gy_abs = vabsq_s16(gy_s);
      
      // Combine Gx and Gy: sobel = |Gx| + |Gy|
      // Convert back to unsigned for addition
      uint16x8_t sobel = vaddq_u16(vreinterpretq_u16_s16(gx_abs), vreinterpretq_u16_s16(gy_abs));
      
      // Saturate to 255
      uint8x8_t result = vqmovn_u16(sobel);
      
      // Store result
      vst1_u8(&out_data[IMG_WIDTH*i + j], result);
    }
    
    for (; j < img_gray.cols - 1; j++) {
      // Calculate Gx
      int gx = abs(
        img_data[IMG_WIDTH*(i-1) + (j-1)] - img_data[IMG_WIDTH*(i-1) + (j+1)] +
        2*img_data[IMG_WIDTH*i + (j-1)] - 2*img_data[IMG_WIDTH*i + (j+1)] +
        img_data[IMG_WIDTH*(i+1) + (j-1)] - img_data[IMG_WIDTH*(i+1) + (j+1)]
      );
      // Calculate Gy
      int gy = abs(
        img_data[IMG_WIDTH*(i-1) + (j-1)] - img_data[IMG_WIDTH*(i+1) + (j-1)] +
        2*img_data[IMG_WIDTH*(i-1) + j] - 2*img_data[IMG_WIDTH*(i+1) + j] +
        img_data[IMG_WIDTH*(i-1) + (j+1)] - img_data[IMG_WIDTH*(i+1) + (j+1)]
      );
      
      // Combine and saturate
      int sobel = gx + gy;
      sobel = (sobel > 255) ? 255 : sobel;
      out_data[IMG_WIDTH*i + j] = sobel;
    }
  }
}