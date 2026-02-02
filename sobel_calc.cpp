#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
#include <arm_neon.h>
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
// void grayScale(Mat& img, Mat& img_gray_out)
// {
//   double color;

//   // #define IMG_WIDTH 640
//   // #define IMG_HEIGHT 480
//   // gray = 0.299*R + 0.587*G + 0.114*B

//   // Convert to grayscale
//   for (int i=0; i<img.rows; i++) {
//     for (int j=0; j<img.cols; j++) {
//       color = .114*img.data[STEP0*i + STEP1*j] +
//               .587*img.data[STEP0*i + STEP1*j + 1] +
//               .299*img.data[STEP0*i + STEP1*j + 2];
//       img_gray_out.data[IMG_WIDTH*i + j] = color;
//     }
//   }
//   // this comes out to 307,200. divided by 16 is 0. so traversing will not leave us out of bounds
//   int total_pixel = img.rows * img.cols; 

//   for (int i = 0; total_pixel; i += 16) {
//     //load into neon vector

//     //compute the math

//     // store
//   }
// }

void grayScale(Mat& img, Mat& img_gray_out)
{
    int total_pixel = img.rows * img.cols; 
    uint8_t* src_ptr = img.data;
    uint8_t* dst_ptr = img_gray_out.data;

    // Define weights as 8-bit constants (scaled by 256)
    // We use uint8 lanes but will widen to 16-bit for math
    uint8x8_t w_blue  = vdup_n_u8(29);  // 0.114 * 256
    uint8x8_t w_green = vdup_n_u8(150); // 0.587 * 256
    uint8x8_t w_red   = vdup_n_u8(77);  // 0.299 * 256

    for (int i = 0; i < total_pixel; i += 16) {
        // 1. Load 16 pixels (48 bytes: BGRBGR...)
        // vld3q_u8 de-interleaves into 3 registers (16 bytes each)
        uint8x16x3_t v_img = vld3q_u8(src_ptr + (i * 3));

        // 2. Process first 8 pixels (Low part)
        // Widen R, G, B to 16-bit and multiply by weights
        uint16x8_t blue_low  = vmull_u8(vget_low_u8(v_img.val[0]), w_blue);
        uint16x8_t green_low = vmull_u8(vget_low_u8(v_img.val[1]), w_green);
        uint16x8_t red_low   = vmull_u8(vget_low_u8(v_img.val[2]), w_red);

        // Sum them: Blue + Green + Red
        uint16x8_t gray_low = vaddq_u16(blue_low, vaddq_u16(green_low, red_low));
        
        // 3. Process next 8 pixels (High part)
        uint16x8_t blue_high  = vmull_u8(vget_high_u8(v_img.val[0]), w_blue);
        uint16x8_t green_high = vmull_u8(vget_high_u8(v_img.val[1]), w_green);
        uint16x8_t red_high   = vmull_u8(vget_high_u8(v_img.val[2]), w_red);

        uint16x8_t gray_high = vaddq_u16(blue_high, vaddq_u16(green_high, red_high));

        // 4. Narrow back down to 8-bit
        // vshrn_n_u16 shifts right by 8 (dividing by 256) and narrows to 8-bit
        uint8x8_t res_low  = vshrn_n_u16(gray_low, 8);
        uint8x8_t res_high = vshrn_n_u16(gray_high, 8);

        // 5. Store 16 finished grayscale pixels
        uint8x16_t final_gray = vcombine_u8(res_low, res_high);
        vst1q_u8(dst_ptr + i, final_gray);
    }
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
void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
{
  Mat img_outx = img_gray.clone();
  Mat img_outy = img_gray.clone();

  // Apply Sobel filter to black & white image
  unsigned short sobel;

  // Calculate the x convolution
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
      sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

      sobel = (sobel > 255) ? 255 : sobel;
      img_outx.data[IMG_WIDTH*(i) + (j)] = sobel;
    }
  }

  // Calc the y convolution
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
     sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

     sobel = (sobel > 255) ? 255 : sobel;

     img_outy.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }

  // Combine the two convolutions into the output image
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
      sobel = img_outx.data[IMG_WIDTH*(i) + j] +
	img_outy.data[IMG_WIDTH*(i) + j];
      sobel = (sobel > 255) ? 255 : sobel;
      img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }
}
