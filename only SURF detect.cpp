#include "stdafx.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "conio.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include <opencv2/nonfree/nonfree.hpp>

#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main()
{
	Mat srcImage = imread("leftmain.jpg", IMREAD_GRAYSCALE);
	Mat closrcImage;

	if( srcImage.empty() )
		return -1;

	cout << srcImage.size() <<endl;
	Mat imageROI=srcImage(Rect(1100,500,900,1100));
	imageROI.copyTo(closrcImage);
	cvtColor(closrcImage,closrcImage,CV_GRAY2BGR);


	//threshold(imageROI,imageROI,110,255,THRESH_BINARY);
//	Mat srcImage = imread("book.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat clothrimage;
	resize(imageROI,clothrimage,Size(1000,800));
	imshow("clothrimage",clothrimage);
	waitKey();
 
	vector<KeyPoint> keypoints;
    Mat descriptors;
 //OpenCV2.4.10 
	//SurfFeatureDetector  surF(3000);
	//surF.detect(imageROI,  keypoints);

	SIFT  siftF(5000, 5);
	siftF.detect(imageROI,  keypoints);


	KeyPointsFilter::retainBest( keypoints,5000);
	cout << "keypoints.size()=" <<  keypoints.size() << endl;
	 
//	SurfDescriptorExtractor extractor; 
//	extractor.compute(srcImage, keypoints, descriptors);
	//surF.compute(srcImage, keypoints, descriptors);
	siftF.compute(srcImage, keypoints, descriptors);

 	FileStorage fs("Keypoints.yml", FileStorage::WRITE);
	write(fs, "keypoints", keypoints);
	write(fs, "descriptors", descriptors);
	fs.release();

	Mat dstImage(srcImage.size(), CV_8UC3);
 	cvtColor(srcImage, dstImage, COLOR_GRAY2BGR);
	drawKeypoints(srcImage, keypoints, dstImage);

	KeyPoint element;
	cout<<"size : "<<keypoints.size()<<endl;
	for(int k=0; k< keypoints.size(); k++)
	{
		element = keypoints[k];
 		RotatedRect rRect = RotatedRect(element.pt, 
			                   Size2f(element.size, element.size), element.angle);
		Point2f vertices[4];
		rRect.points(vertices);
		for (int i = 0; i < 4; i++)
			line(dstImage, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 2);

		circle(closrcImage, element.pt, cvRound(element.size/2), 
			             Scalar(rand()%256,rand()%256,rand()%256), 2);
		cout << element.size/2 <<endl;
		//Mat closrc;
		//resize(closrcImage,closrc,Size(1000,800));
		//imshow("thr_imag",closrcImage);
		//waitKey();
	}               
	resize(closrcImage,closrcImage,Size(1000,800));
	imshow("thr_imag",closrcImage);
	waitKey();
	return 0;
}

