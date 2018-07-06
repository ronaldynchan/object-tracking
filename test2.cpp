/******************************************************************************
Filename:		ball_tracking.cpp
Author:			Ronald Chan (adapted from Python script also by Ronald Chan
				with parts inspired by A. Rosebrock [REF] and  W. Lucetti [REF])
Project:		Degree Project
Description:	Script using OpenCV API to detect and track ball based on the 
				ball's color.

rev.5: 	working distance measurement and object detection
rev.6: 	added fps count
rev.7: 	added Kalman filtering to improve tracking, major rework done to 
		accommodate new features
rev.8: 	fixed bugs, improved speed 
rev.9: 	added headers and code to write to FPGA fabric
******************************************************************************/
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/video/video.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <sys/time.h>

#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include "hps_0.h"

using namespace std;
using namespace cv;

//function declarations
void handler(int signo);
float initialize(UMat frame);

//global variables
UMat frame, dummy_frame;
float focal_length;

//range of colors to track
#define HSV_lower Scalar(5,0,210)						
#define HSV_upper Scalar(40,255,255)

//known diameter of ball
#define known_diameter 0.0381

//approximate distance from camera to object
#define known_distance 0.6858

//set size of captured frame
#define new_width 320
#define new_height 240

//(LW H2F Bridge address = 0xff200000) (H2F Bridge address = 0xC0000000)
#define REG_BASE 0xff200000 
//(LW H2F Bridge Span = 0x00200000) (H2F Bridge Span = 0x3C000000)
#define REG_SPAN 0x00200000 

//addresses and variables used for memory mapping
void *base;
uint32_t *servo_x_act, *servo_x_pred, *servo_y_act, *servo_y_pred;
int fd;

/******************************************************************************
function:	mai
******************************************************************************/
int main(void) { 
	
	// >>>>>> Memory mapping
	fd = open("/dev/mem", O_RDWR|O_SYNC);
	if(fd<0)
	{
		printf("Can't open memory .\n");
		return -1;
	}
	//calculate the base address
	base = mmap(NULL, REG_SPAN, PROT_READ|PROT_WRITE, MAP_SHARED, fd, REG_BASE);
	if(base == MAP_FAILED)
	{
		printf("Can't map memory. \n");
		close(fd);
		return -1;
	}
	//calculate the memory address of the four variables 
	servo_x_act = (uint32_t*)(base + SERVO_ACT_X_BASE);
	servo_x_pred = (uint32_t*)(base + SERVO_PRED_X_BASE);
	servo_y_act = (uint32_t*)(base + SERVO_ACT_Y_BASE);
	servo_y_pred = (uint32_t*)(base + SERVO_PRED_Y_BASE);
	signal(SIGINT, handler);
	// <<<<<< Memory mapping
	
	
	// >>>>>> Kalman filter
    int stateSize = 6; //number of state variables
    int measSize = 4; //number of measurement variables
    int contrSize = 0;	//size of control vector
	unsigned int type = CV_32F;	

	//instantiate a Kalman filter with cv::KalmanFilter
    KalmanFilter kf(stateSize, measSize, contrSize, type); 

    Mat state(stateSize, 1, type);  //column vector [x ,y, v_x, v_y, w, h]'
    Mat meas(measSize, 1, type);    //column vector [z_x, z_y, z_w, z_h]'
    
    //transition State Matrix A
    //note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
	//first create a 6x6 identity matrix --> dTs added on second measurement
    setIdentity(kf.transitionMatrix); 

    //measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;
	
	//control matrix B
	// [ 1 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 ]
 	// [ 0 0 1 0 0 0 ]
	// [ 0 0 0 1 0 0 ]
	// [ 0 0 0 0 0 0 ]
	// [ 0 0 0 0 0 0 ]
	//setIdentity(kf.controlMatrix);
	//kf.controlMatrix.at<float>(0) = 1.0f;
	//kf.controlMatrix.at<float>(7) = 1.0f;
	//kf.controlMatrix.at<float>(14) = 1.0f;
	//kf.controlMatrix.at<float>(21) = 1.0f;

    //Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    kf.processNoiseCov.at<float>(0) = 1e-3;
    kf.processNoiseCov.at<float>(7) = 1e-3;
    kf.processNoiseCov.at<float>(14) = 1e-1;
    kf.processNoiseCov.at<float>(21) = 1e-1;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    //measures noise covariance matrix R
    setIdentity(kf.measurementNoiseCov, Scalar(1e-1)); 
    // <<<<<< Kalman Filter
	
	
	// >>>>>> Setup
	//create videoCapture object
	VideoCapture capture; 
	
	//initialize window to display captured video 
	namedWindow("ball tracker + distance detection", CV_WINDOW_AUTOSIZE); 
	
	//start the video stream from web cam
	capture.open(-1); //open the first available camera
	
	//error handling for capture failure
	if(!capture.isOpened()) 
	{
		printf("Failed to access webcam\n"); 
		return -1;
	}
	
	//read the first captured frame and initialize the system
	capture.read(frame);
	
	//set captured frame size
	capture.set(CV_CAP_PROP_FRAME_WIDTH, new_width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, new_height);
	
	//call initialize function to grab focal length
	focal_length = initialize(frame); 
	
	//ball tracking setup before main loop
	double ticks = 0;
	bool found = false;
	bool measure_distance = false; //do not start with distance measurement
	bool center_select = false; //switch between predicted and actual centers
	int largest_area;
	bool is_square; 
	bool show_thresholding = false;
	bool use_kalman = false;	

	//fps count and elapsed time setup
	char str_time[50];
	static struct timeval last_time;
	struct timeval current_time;
	static float last_fps;
	float t;
	float fps;
	float elapsed_time = 0.0;
	// <<<<<< Setup
	
	
	// >>>>>> main loop
	while(capture.read(frame)) 
	{
		//error handling check for no frame captured
		if(frame.empty()) 
		{
			printf("No captured frame -- Break!"); 
			break;
		}	
			
		//find t-t0
        double precTick = ticks;
        ticks = (double) getTickCount();
        double dT = (ticks - precTick) / getTickFrequency(); //seconds
		
		Mat new_frame;
		frame.copyTo(new_frame);
		
		
		// >>>>>> State prediction
		if (found) //should be true on second captured frame
		{
            //update matrix A 
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;

            //cout << "dT:" << endl << dT << endl;

            state = kf.predict(); //predict location
            //cout << "State post:" << endl << state << endl;
			
			//make a rectangle object with width, height and (x,y)
            Rect predicted_rect;	
            predicted_rect.width = state.at<float>(4);
            predicted_rect.height = state.at<float>(5);
            predicted_rect.x = state.at<float>(0) - predicted_rect.width / 2;
            predicted_rect.y = state.at<float>(1) - predicted_rect.height / 2;
			
			//predicted center coordinates
			Point predicted_center;
            predicted_center.x = state.at<float>(0);
            predicted_center.y = state.at<float>(1);
			
			
			//select between actual and predicted coordinates output
			//if there is no previous actual coordinate, use predicted
			if (center_select && use_kalman) 
			{  
				//place a blue dot at predicted center of ball
				circle(new_frame, predicted_center, 2, CV_RGB(0,0,255), -1); 
			
				//put a blue rectangle around predicted ball location
				rectangle(new_frame, predicted_rect, CV_RGB(0,0,255), 2); 
			
				//label predicted location at bottom right of outline
				putText(new_frame, "predicted", 
				Point(predicted_center.x + (predicted_rect.width / 2) + 5, 
					predicted_center.y + (predicted_rect.height / 2)),
				FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,0,255), 1);
				
				//assign predicted coordinates to memory addresses
				*servo_x_pred = predicted_center.x; 
				*servo_y_pred = predicted_center.y;
				cout << "x pred: " << *servo_x_pred << endl;
				cout << "y pred: " << *servo_y_pred << endl;
				//display predicted center coordinates onto frame
				stringstream sstr_pred;
				sstr_pred << "(" << predicted_center.x << "," << predicted_center.y << ")";
				putText(new_frame, sstr_pred.str(),
                        Point(predicted_center.x, predicted_center.y + 5),
                        FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,0,255), 1);
			}
			else 
			{
				*servo_x_pred = 0x0; 
				*servo_y_pred = 0x0;
				cout << "x pred: " << *servo_x_pred << endl;
				cout << "y pred: " << *servo_y_pred << endl;
			}
        }
		// <<<<<< State prediction
		
		
		// >>>>>> Thresholding
		//smooth out noise with 5x5 Gaussian kernel, sigma X & Y are 3.0
		UMat blur;
		GaussianBlur(frame, blur, Size(5,5), 3.0, 3.0); 
		
		//change to HSV color space
		UMat frame_hsv;
		cvtColor(frame, frame_hsv, COLOR_BGR2HSV); 
		
		//threshold for desired color range
		UMat mask;
		inRange(frame_hsv, HSV_lower, HSV_upper, mask); 
			
		//smooth edges with morphological operations to remove noise and isolate contours							
		erode(mask, mask, Mat(), Point(-1,-1), 2); 		
		dilate(mask, mask, Mat(), Point(-1,-1), 2);
		
		//view thresholding
		if (show_thresholding == true)
        		imshow("Threshold", mask);
		// <<<<<< Thresholding
		
		
		// >>>>>> Object detection
		//find all contours in thresholded image
		vector<vector<Point>> contours;
	
		findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		
		largest_area = 0;
	
		//if there are any contours found, search for the largest circular contour
		Rect bBox;
		
		if(contours.size()>0) 
		{									
			//iterate through each contour
			for(size_t i=0; i<abs(contours.size()); i++) 
			{		
				is_square = false; //reset "squareness" check
				double area = contourArea(contours[i]); //find area of each contour
				//determine the largest contour in terms of area
				if(area>largest_area) 
				{
					largest_area = area;
				}
				
				//check for "roundness"
				bBox = boundingRect(contours[i]); //returns top left vertex, width and height
				
				float ratio = (float) bBox.width / (float) bBox.height;
				if (ratio > 1.0f)
					ratio = 1.0f / ratio;
				
				// Searching for a bBox almost square
				if (ratio > 0.75)
				{
					is_square = true;
				}
			}
			
			//only proceed if radius meets minimum value and bounding rectangle is square
			if(bBox.width > 1 && is_square)  
			{
				Point actual_center;
				actual_center.x = bBox.x + bBox.width / 2; //pixel (0,0) is top-left vertex
				actual_center.y = bBox.y + bBox.height / 2;
				
				//draw red dot in center of actual ball location
				circle(new_frame, actual_center, 2, CV_RGB(255,0,0), -1);
				
				//draw box around actual ball location
				rectangle(new_frame, bBox, CV_RGB(255,0,0), 2);
				
				//label the measured object
				putText( new_frame, "actual", 
					Point(actual_center.x + (bBox.width / 2) + 5, 
						actual_center.y - bBox.height / 2), 
					FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1);
					
				//show actual center coordinates
				stringstream sstr_act;
				sstr_act << "(" << actual_center.x << "," << actual_center.y << ")";
				putText(new_frame, sstr_act.str(),
                        Point(actual_center.x, actual_center.y + 5),
                        FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1);			
		// <<<<<< Object detection
		
		
				// >>>>>> Distance measurement
				//calculate distance and display it in bottom right corner of frame
				float inches = ( known_diameter * focal_length ) / bBox.width; 
				
				if(measure_distance == true) 
				{	
					char str_dist[50];
					sprintf(str_dist, "%.2f m", inches); 
					
					//display calculated distance to object
					
					putText(new_frame, str_dist, 
					Point(bBox.x, bBox.y + bBox.height + 20), 
					FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0,255,0), 2);		
				} 
				// <<<<<< Distance measurement
				
				
				// >>>>>> Kalman update
				//update Z matrix with new measurements
				meas.at<float>(0) = actual_center.x;   
				meas.at<float>(1) = actual_center.y;
				meas.at<float>(2) = (float)bBox.width; 
				meas.at<float>(3) = (float)bBox.height;
				
				//if there is actual object location found, don't use predicted values
				center_select = false; 
				*servo_x_act = actual_center.x;
				*servo_y_act = actual_center.y;
				cout << "x act: " << *servo_x_act << endl;
				cout << "y act: " << *servo_y_act << endl;
								
				if (!found) // First detection!
				{
					// >>>> Initialization
					kf.errorCovPre.at<float>(0) = 1; // px
					kf.errorCovPre.at<float>(7) = 1; // px
					kf.errorCovPre.at<float>(14) = 1;
					kf.errorCovPre.at<float>(21) = 1;
					kf.errorCovPre.at<float>(28) = 1; // px
					kf.errorCovPre.at<float>(35) = 1; // px

					state.at<float>(0) = meas.at<float>(0);
					state.at<float>(1) = meas.at<float>(1);
					state.at<float>(2) = 0; //velocity cannot be inferred by one frame!!!
					state.at<float>(3) = 0;	
					state.at<float>(4) = meas.at<float>(2);
					state.at<float>(5) = meas.at<float>(3);
					// <<<< Initialization

					kf.statePost = state;
					
					found = true;
					
				}
				else
					kf.correct(meas); // Kalman Correction
				
				cout << "Measure matrix:" << endl << meas << endl;
				// <<<<<< Kalman update
				
				
			}
			else
			{
				//if no actual object location found, use predicted location
				center_select = true; 
				*servo_x_act = 0x0;
				*servo_y_act = 0x0;
				cout << "x act: " << *servo_x_act << endl;
				cout << "y act: " << *servo_y_act << endl;
			}
		}
		
		else 
			found = false; //object not found, Kalman not updated
					
		//display fps count in bottom left corner of frame
		gettimeofday(&current_time, NULL);
		t = (current_time.tv_sec - last_time.tv_sec) + 
			(current_time.tv_usec - last_time.tv_usec) / 1000000.;
		fps = 1. / t;
		fps = last_fps * 0.8 + fps * 0.2;
		last_fps = fps;
		last_time = current_time;
		sprintf(str_time, "%2.2f", fps);
		putText(new_frame, str_time, Point(5, new_height-5), 
			FONT_HERSHEY_DUPLEX, 1, CV_RGB(255,0,0));
		
		//display frame 
		imshow("ball tracker + distance detection", new_frame);	
		
		//Check for key presses
		int c = waitKey(60); //check for key press every 60ms
		if (c == 27 || c == 'q' || c == 'Q') break;	//quit when q or esc pressed
		if (c == 100) measure_distance = !measure_distance; //d to toggle distance measurement
		if (c == 116) show_thresholding = !show_thresholding; //t to show thresholding
		if (c == 107) use_kalman = !use_kalman; //k to toggle Kalman filtering
	}
	// <<<<<< main loop
	
	//clean up
	capture.release();			
	destroyAllWindows();		
    return EXIT_SUCCESS;		
}		
/******************************************************************************		
function:	 initialize
description: From the first frame, determine the focal length. 
******************************************************************************/
float initialize(UMat frame) {	
	
	//smooth out noise (Gaussian) with 5x5 Gaussian kernel, sigma X & Y are 3.0
	UMat blur;
	GaussianBlur(frame, blur, Size(5,5), 3.0, 3.0); 
	
	//change to HSV color space
	UMat frame_hsv;
	cvtColor(blur, frame_hsv, COLOR_BGR2HSV); 
	
	//threshold for desired color range
	UMat mask;
	inRange(frame_hsv, HSV_lower, HSV_upper, mask); 
		
	//smooth edges with morphological operations to remove noise and isolate contours							
	erode(mask, mask, Mat(), Point(-1,-1), 2); 		
	dilate(mask, mask, Mat(), Point(-1,-1), 2);

	vector<vector<Point>> contours;
	bool is_square;
	int largest_area = 0;
	
	//Find all contours in thresholded image and determine the largest one
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); 
	
	//if there are any contours found
	if(contours.size()>0) 
	{		
		vector<cv::Rect> ballsBox;
		
		//iterate through each contour
		for(size_t i=0; i<abs(contours.size()); i++) 		
		{
			is_square = false; //reset "squareness" check
			double area=contourArea(contours[i]); //find area of each contour, store in vector area
			if(area>largest_area) 
			{
				largest_area = area;
			}
			Rect bBox;
			bBox = boundingRect(contours[i]); //returns top left vertex, width and height
			
			float ratio = (float) bBox.width / (float) bBox.height;
			if (ratio > 1.0f)
				ratio = 1.0f / ratio;
			
			//searching for a bBox almost square
			if (ratio > 0.75)
			{
				is_square = true;
				ballsBox.push_back(bBox);
			}
		}
		
		//calculate the focal length and return the value
		//only proceed if radius meets minimum value and bounding rectangle is square
		if(ballsBox[0].width > 1 && is_square)  	
		{
			//f = diameter_in_pixels * distance_in_inches / diameter_in_inches
			focal_length = (ballsBox[0].width * known_distance) / (2*known_diameter);				
			cout << "focal length: " << focal_length << endl;
			if(focal_length>0) return focal_length; //return non-negative value
			}
	}
}
/******************************************************************************
function:	 handler
description: initialize addresses for memory mapping
******************************************************************************/
void handler(int signo)
{
	*servo_x_act = 0;
	*servo_x_pred = 0;
	*servo_y_act = 0;
	*servo_y_pred = 0;
	munmap(base, REG_SPAN);
	close(fd);
	exit(0);
}