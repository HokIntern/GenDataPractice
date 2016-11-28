#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

using namespace std;

const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

int DELAY_CAPTION = 0;
int DELAY_BLUR = 300;

cv::Mat src; 
cv::Mat grayscale; 
cv::Mat gaussianBlur;
cv::Mat adaptiveThreshold;
std::vector<std::vector<cv::Point> > contours;
std::vector<cv::Vec4i> v4iHierarchy;
cv::Mat dst;
char window_name[] = "OCR Steps";

/// Function headers
int display_caption(string caption);
int display_dst(int delay);

int main() {
	
	src = cv::imread("E:\\downloads\\dashcamtest2_timestamp.png", 1);

	if (display_caption("Original Image") != 0) { return 0; }

	dst = src.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }

	cv::cvtColor(
		src, 
		grayscale, 
		CV_BGR2GRAY
	);
	if (display_caption("Grayscale Image") != 0) { return 0; }
	dst = grayscale.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }

	cv::GaussianBlur(
		grayscale, 
		gaussianBlur, 
		cv::Size(5, 5), 
		0
	);
	if (display_caption("Gaussian Blur Image") != 0) { return 0; }
	dst = gaussianBlur.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }

	cv::adaptiveThreshold(
		gaussianBlur,
		adaptiveThreshold,
		255,
		CV_ADAPTIVE_THRESH_GAUSSIAN_C,
		CV_THRESH_BINARY,
		11,
		5
	);

	
	string caption = "b: " + to_string(5) + ", C: " + to_string(3) + "\nAdaptive";
	//if (display_caption(caption) != 0) { return 0; }
	dst = adaptiveThreshold.clone();
	if (display_dst(DELAY_CAPTION) != 0) { return 0; }
	if (display_caption(caption) != 0) { return 0; }
	
	
	/*
	for (int bSize = 3; bSize < 20; bSize += 2) {
		for (int C = 0; C < 6; C++) {
			cv::adaptiveThreshold(
				gaussianBlur,
				adaptiveThreshold,
				255,
				CV_ADAPTIVE_THRESH_GAUSSIAN_C,
				CV_THRESH_BINARY,
				bSize,
				C
			);

			string caption = "b: " + to_string(bSize) + ", C: " + to_string(C) + "\nAdaptive";
			//if (display_caption(caption) != 0) { return 0; }
			dst = adaptiveThreshold.clone();
			if (display_dst(DELAY_CAPTION) != 0) { return 0; }
			if (display_caption(caption) != 0) { return 0; }
		}
	}
	*/
	
	cv::findContours(adaptiveThreshold,
		contours,
		v4iHierarchy,
		CV_RETR_LIST,
		CV_CHAIN_APPROX_TC89_KCOS
	);

	for (int i = 0; i < contours.size(); i++) {
		if (cv::contourArea(contours[i]) > MIN_CONTOUR_AREA) {
			cv::Rect boundingRect = cv::boundingRect(contours[i]);

			cv::rectangle(src, boundingRect, cv::Scalar(0, 0, 255), 2);      // draw red rectangle around each contour as we ask user for input

			cv::Mat matROI = adaptiveThreshold(boundingRect);           // get ROI image of bounding rect

			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

			cv::imshow("matROI", matROI);
			cv::imshow("matROIResized", matROIResized);
			cv::imshow("imgTrainingNumbers", src);

			int intChar = cv::waitKey(0);

			if (intChar == 27) { //esc pressed
				return(0);
			}
		}
	}


	/*
	cv::findContours();
	cv::resize();
	*/
}

int display_caption(string caption)
{
	cv::Mat captionMat = cv::Mat::zeros(src.size(), src.type());
	putText(captionMat, caption,
		cv::Point(src.cols / 4, src.rows / 2),
		CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 255, 255));

	cv::imshow(window_name, captionMat);
	int c = cv::waitKey(DELAY_CAPTION);
	if (c == 27) { return -1; } //esc pressed
	return 0;
}

int display_dst(int delay)
{
	cv::imshow(window_name, dst);
	int c = cv::waitKey(delay);
	if (c == 27) { return -1; } //esc pressed
	return 0;
}