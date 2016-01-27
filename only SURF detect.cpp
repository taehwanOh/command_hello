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
	Mat srcImage = imread("image.jpg", IMREAD_GRAYSCALE);
	Mat closrcImage;

	if( srcImage.empty() )
		return -1;

	cout << srcImage.size() <<endl;
	Mat imageROI=srcImage(Rect(1100,500,900,1100));
	imageROI.copyTo(closrcImage);
	cvtColor(closrcImage,closrcImage,CV_GRAY2BGR);

	vector<KeyPoint> keypoints;
    Mat descriptors;
 //OpenCV2.4.10 
	//SurfFeatureDetector  surF(10);
	//surF.detect(imageROI,  keypoints);

	SIFT  siftF(10000, 3); //n개의 포인트 detect
	siftF.detect(imageROI,  keypoints);


	//KeyPointsFilter::retainBest( keypoints,1000); //retainbest 로 n개를 걸러냄
	cout << "keypoints.size()=" <<  keypoints.size() << endl;
	 
//	SurfDescriptorExtractor extractor; 
//	extractor.compute(srcImage, keypoints, descriptors);
	//surF.compute(srcImage, keypoints, descriptors);
	siftF.compute(imageROI, keypoints, descriptors);

 	FileStorage fs("Keypoints.yml", FileStorage::WRITE);
	write(fs, "keypoints", keypoints);
	write(fs, "descriptors", descriptors);
	fs.release();

	Mat dstImage(srcImage.size(), CV_8UC3);
 	cvtColor(srcImage, dstImage, COLOR_GRAY2BGR);
	drawKeypoints(srcImage, keypoints, dstImage);

	cout<<"size : "<<keypoints.size()<<endl;

	//Mat cloimage;
	//srcImage.copyTo(cloimage);

	Mat thrImage;
	threshold(imageROI,thrImage, 120, 255, THRESH_BINARY_INV);
	//Mat rthrImage;
	//resize(thrImage,rthrImage,Size(1000,800));
	//imshow("thr_imag",rthrImage);
	//waitKey();

	Mat copyimg;
	closrcImage.copyTo(copyimg);

	//for(int k=0; k< keypoints.size(); k++)
	//{
	//	KeyPoint element;
	//	element = keypoints[k];
	//	circle(copyimg, element.pt, element.size,Scalar(0,0,255), 2);
	//
	//	//Mat clos;
	//	//copyimg.copyTo(clos);
	//	//circle(clos, element.pt, 2,Scalar(0,0,255), 2);
	//	//resize(clos,clos,Size(1000,800));
	//	//imshow("clos",clos);
	//	//waitKey();
	//
	//
	//}
	//	
	//resize(copyimg,copyimg,Size(1000,800));
	//imshow("clos",copyimg);
	//waitKey();
	
	//Filter1 , threshold 후 3*3 kernel에 흰색이 모두 차있을 경우 point로 지정
	vector <KeyPoint> keypoints_1;

	for(int k=0; k< keypoints.size(); k++)
	{
		KeyPoint element;
		element = keypoints[k];

		double key_x = element.pt.x-1;
		double key_y = element.pt.y-1;
	
		if(key_x<0)
			key_x = 0;
	
		if(key_y<0)
			key_y = 0;
	
		if(key_x+3> imageROI.cols)
			key_x = imageROI.cols-3;
	
		if(key_y+3> imageROI.rows)
			key_y = imageROI.rows-3;
	
	
		Mat kernel = thrImage(Rect(key_x,key_y,3,3));
		
		//circle(closrcImage, element.pt, cvRound(element.size/2), 
		//	             Scalar(rand()%256,rand()%256,rand()%256), 2);

		
		int nwhite=0;
		for(int p=0;p<3;p++)
		{
			for(int q=0;q<3;q++)
			{
				if(float(kernel.at<uchar>(p,q))==255)
				{
					nwhite+=1;
				}
			}
		}

		//Mat cloim;
		//closrcImage.copyTo(cloim);
		//rectangle(cloim,Rect(key_x,key_y,3,3),Scalar(0,0,255),1);
		//resize(cloim,cloim,Size(1000,800));
		//imshow("thr_imag",cloim);
		//waitKey();

		//  && element.response >= 0.04
		if(nwhite ==9)
			keypoints_1.push_back(element);
	}  

	cout << keypoints_1.size() <<endl;
	for(int k=0; k< keypoints_1.size(); k++)
	{
		KeyPoint element;
		
		element = keypoints_1[k];
		//cout << element.response <<endl;
		circle(closrcImage, element.pt, 2, 
			             Scalar(0,0,255), 2);
		//Mat closrcImage2;
		//closrcImage.copyTo(closrcImage2);
		//circle(closrcImage2, element.pt, 2, 
		//	             Scalar(0,0,255), 2);
		//resize(closrcImage2,closrcImage2,Size(1000,800));
		//imshow("final",closrcImage2);
		//waitKey();
	}

	resize(closrcImage,closrcImage,Size(1000,800));
	imshow("final",closrcImage);
	waitKey();

	//Filter2, 
	vector <KeyPoint> keypoints_2;

	resize(closrcImage,closrcImage,Size(1000,800));
	imshow("final",closrcImage);
	waitKey();
	return 0;
}

