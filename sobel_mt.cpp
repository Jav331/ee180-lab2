#include <stdio.h>
#include <stdlib.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include <sys/ioctl.h>
#include <err.h>

#include "sobel_alg.h"
#include "pc.h"

// Replaces img.step[0] and img.step[1] calls in sobel calc

using namespace cv;

static ofstream results_file;

// Define image mats to pass between function calls
static Mat img_gray, img_sobel;
static float total_fps, total_ipc, total_epf;
static float gray_total, sobel_total, cap_total, disp_total;
static float sobel_ic_total, sobel_l1cm_total;

// Global synchronization primitives
static pthread_barrier_t barrier_cap, barrier_gray, barrier_sobel, barrier_loop;
static pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile bool keep_running = true; // so that threads can know when to exit loop
// Src global so both threads can access
static Mat src;

/*******************************************
 * Model: runSobelMT
 * Input: None
 * Output: None
 * Desc: This method pulls in an image from the webcam, feeds it into the
 * sobelCalc module, and displays the returned Sobel filtered image. This
 * function processes NUM_ITER frames.
 ********************************************/
void *runSobelMT(void *ptr)
{
  // Set up variables for computing Sobel
  string top = "Sobel Top";
  uint64_t cap_time, gray_time, sobel_time, disp_time, sobel_l1cm, sobel_ic;
  pthread_t myID = pthread_self();
  counters_t perf_counters;

  // Local counters
  uint64_t t_l1cm = 0, t_ic = 0;

  // Allow the threads to contest for thread0 (controller thread) status
  pthread_mutex_lock(&thread0);

  // Check to see if this thread is first to this part of the code
  if (thread0_id == 0) {
    thread0_id = myID;
    // Initialize barriers for 2 threads
    pthread_barrier_init(&barrier_cap, NULL, 2);
    pthread_barrier_init(&barrier_gray, NULL, 2);
    pthread_barrier_init(&barrier_sobel, NULL, 2);
    pthread_barrier_init(&barrier_loop, NULL, 2);
  }
  pthread_mutex_unlock(&thread0);

  pc_init(&perf_counters, 0);

  // Start algorithm
  CvCapture* video_cap = NULL;

  // Only thread 0 initializes the camera
  if (myID == thread0_id) {
      if (opts.webcam) {
        video_cap = cvCreateCameraCapture(-1);
      } else {
        video_cap = cvCreateFileCapture(opts.videoFile);
      }
      
      cvSetCaptureProperty(video_cap, CV_CAP_PROP_FRAME_WIDTH, IMG_WIDTH);
      cvSetCaptureProperty(video_cap, CV_CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT);
  }

  // Keep track of the frames
  int i = 0;

  // Loop condition handles exit thread
  while (keep_running) {
    
    // Capture
    if (myID == thread0_id) {
        // Allocate memory to hold grayscale and sobel images
        img_gray = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);
        img_sobel = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);

        pc_start(&perf_counters);
        src = cvQueryFrame(video_cap);
        pc_stop(&perf_counters);

        cap_time = perf_counters.cycles.count;
        sobel_l1cm = perf_counters.l1_misses.count;
        sobel_ic = perf_counters.ic.count;
    }

    // Wait for capture
    pthread_barrier_wait(&barrier_cap);

    // Check capture success and loop finished
    if (myID == thread0_id && src.empty()) {
        keep_running = false;
    }

    // Sync threads on exit condition
    pthread_barrier_wait(&barrier_loop); 
    if (!keep_running) break;

    // LAB 2, PART 2: Start parallel section
    
    // ROIs for parallel processing
    int split_row = IMG_HEIGHT / 2;
    Mat src_roi, gray_roi, sobel_input_roi, sobel_output_roi;

    if (myID == thread0_id) {
        // Top half
        src_roi = src(Rect(0, 0, IMG_WIDTH, split_row));
        gray_roi = img_gray(Rect(0, 0, IMG_WIDTH, split_row));
        // Overlap for Sobel stencil (+1 row at bottom)
        sobel_input_roi = img_gray(Rect(0, 0, IMG_WIDTH, split_row + 1));
        sobel_output_roi = img_sobel(Rect(0, 0, IMG_WIDTH, split_row + 1));
    } else {
        // Bottom half
        src_roi = src(Rect(0, split_row, IMG_WIDTH, IMG_HEIGHT - split_row));
        gray_roi = img_gray(Rect(0, split_row, IMG_WIDTH, IMG_HEIGHT - split_row));
        // Overlap for Sobel stencil (-1 row at top)
        sobel_input_roi = img_gray(Rect(0, split_row - 1, IMG_WIDTH, IMG_HEIGHT - split_row + 1));
        sobel_output_roi = img_sobel(Rect(0, split_row - 1, IMG_WIDTH, IMG_HEIGHT - split_row + 1));
    }

    pc_start(&perf_counters);
    grayScale(src_roi, gray_roi); // Pass ROI
    pc_stop(&perf_counters);

    // Local vars for aggregation later
    uint64_t my_gray_cycles = perf_counters.cycles.count;
    t_l1cm = perf_counters.l1_misses.count;
    t_ic = perf_counters.ic.count;

    // Wait for phases
    pthread_barrier_wait(&barrier_gray);

    pc_start(&perf_counters);
    sobelCalc(sobel_input_roi, sobel_output_roi); // Pass ROI
    pc_stop(&perf_counters);

    // Accumulate local vars
    uint64_t my_sobel_cycles = perf_counters.cycles.count;
    t_l1cm += perf_counters.l1_misses.count;
    t_ic += perf_counters.ic.count;
    // LAB 2, PART 2: End parallel section

    // Wait for display
    pthread_barrier_wait(&barrier_sobel);

    // Add stats from both threads to T0 variables
    pthread_mutex_lock(&stats_mutex);
    if (myID != thread0_id) {
        // Worker thread adds its stats to global accumulator
        sobel_l1cm_total += t_l1cm; 
        sobel_ic_total += t_ic;
    }
    static uint64_t t1_l1cm_share = 0;
    static uint64_t t1_ic_share = 0;
    if (myID != thread0_id) {
        t1_l1cm_share = t_l1cm;
        t1_ic_share = t_ic;
    }
    pthread_mutex_unlock(&stats_mutex);
    
    // T0 waits for T1 vars
    pthread_barrier_wait(&barrier_loop); 

    if (myID == thread0_id) {
        // Add T1's stats to T0's local counters
        sobel_l1cm += (t_l1cm + t1_l1cm_share); // T0 capture + T0 processing + T1 processing
        sobel_ic += (t_ic + t1_ic_share);
        
        gray_time = my_gray_cycles; 
        sobel_time = my_sobel_cycles;

        pc_start(&perf_counters);
        namedWindow(top, CV_WINDOW_AUTOSIZE);
        imshow(top, img_sobel);
        pc_stop(&perf_counters);

        disp_time = perf_counters.cycles.count;
        sobel_l1cm += perf_counters.l1_misses.count;
        sobel_ic += perf_counters.ic.count;

        cap_total += cap_time;
        gray_total += gray_time;
        sobel_total += sobel_time;
        sobel_l1cm_total += sobel_l1cm;
        sobel_ic_total += sobel_ic;
        disp_total += disp_time;
        total_fps += PROC_FREQ/float(cap_time + disp_time + gray_time + sobel_time);
        total_ipc += float(sobel_ic/float(cap_time + disp_time + gray_time + sobel_time));
        i++;

        // Press q to exit
        char c = cvWaitKey(10);
        if (c == 'q' || i >= opts.numFrames) {
            keep_running = false;
        }
    }
    
    // Final Sync
    pthread_barrier_wait(&barrier_loop);
    if (!keep_running) break;
  }

  // Thread 0 reports
  if (myID == thread0_id) {
      total_epf = PROC_EPC*NCORES/(total_fps/i);
      float total_time = float(gray_total + sobel_total + cap_total + disp_total);

      results_file.open("mt_perf.csv", ios::out);
      results_file << "Percent of time per function" << endl;
      results_file << "Capture, " << (cap_total/total_time)*100 << "%" << endl;
      results_file << "Grayscale, " << (gray_total/total_time)*100 << "%" << endl;
      results_file << "Sobel, " << (sobel_total/total_time)*100 << "%" << endl;
      results_file << "Display, " << (disp_total/total_time)*100 << "%" << endl;
      results_file << "\nSummary" << endl;
      results_file << "Frames per second, " << total_fps/i << endl;
      results_file << "Cycles per frame, " << total_time/i << endl;
      results_file << "Energy per frames (mJ), " << total_epf*1000 << endl;
      results_file << "Total frames, " << i << endl;
      results_file << "\nHardware Stats (Cap + Gray + Sobel + Display)" << endl;
      results_file << "Instructions per cycle, " << total_ipc/i << endl;
      results_file << "L1 misses per frame, " << sobel_l1cm_total/i << endl;
      results_file << "L1 misses per instruction, " << sobel_l1cm_total/sobel_ic_total << endl;
      results_file << "Instruction count per frame, " << sobel_ic_total/i << endl;

      cvReleaseCapture(&video_cap);
      results_file.close();
  }

  pthread_barrier_wait(&endSobel);
  return NULL;
}