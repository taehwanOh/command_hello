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

void onMouse(int event, int x, int y, int flags, void* param);
void thinningIteration(cv::Mat& im, int iter);
void thinning(cv::Mat& im);

int main()
{
	Mat srcImage = imread("image.jpg", IMREAD_GRAYSCALE); //srcImage : 원본이미지
	Mat mouseimage =srcImage.clone();
	
	imshow("mouseimage",mouseimage);
	setMouseCallback("mouseimage", onMouse,  (void *)&mouseimage);
	//resize(mouseimage,mouseimage,Size(1000,800));
	waitKey();

	int x,y,disx,disy;
	cout << "x : " ;
	cin >> x ;
	cout << endl << "y : ";
	cin >> y;
	cout <<endl<<"disx : " ;
	cin >> disx;
	cout <<endl<<"disy : ";
	cin >> disy;

	if( srcImage.empty() )
		return -1;

	cout << srcImage.size() <<endl;
	Mat imageROI=srcImage(Rect(x,y,disx,disy)); //imageROI : 영역확대이미지


	vector<KeyPoint> keypoints;
 
	//SurfFeatureDetector  surF(10);
	//surF.detect(imageROI,  keypoints);
	SIFT  siftF(10000, 3); //n개의 포인트 detect
	siftF.detect(imageROI,  keypoints);
	//KeyPointsFilter::retainBest( keypoints,1000); //retainbest 로 n개를 걸러냄
	cout << "keypoints.size()=" <<  keypoints.size() << endl;
	 
	Mat descriptors;
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


	Mat copyimg;
	imageROI.copyTo(copyimg);
	cvtColor(copyimg,copyimg,CV_GRAY2BGR);

	for(int k=0; k< keypoints.size(); k++)
	{
		KeyPoint element;
		element = keypoints[k];
		circle(copyimg, element.pt, 2,Scalar(0,0,255), -1);
	
		//Mat clos;
		//copyimg.copyTo(clos);
		//circle(clos, element.pt, 2,Scalar(0,0,255), 2);
		//resize(clos,clos,Size(1000,800));
		//imshow("clos",clos);
		//waitKey();
	
	
	}
		
	resize(copyimg,copyimg,Size(1000,800));
	imshow("clos",copyimg);
	waitKey();
	
	//Filter1 , threshold 후 3*3 kernel에 흰색이 모두 차있을 경우 point로 지정
	
	Mat thrImage;
	threshold(imageROI,thrImage, 120, 255, THRESH_BINARY_INV);
	imwrite("thrImage.jpg",thrImage);

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

		if(nwhite ==9)
			keypoints_1.push_back(element);
	}  

	Mat closrcImage; //closrcImage : circle를 위한 복사이미지
	imageROI.copyTo(closrcImage); 

	cvtColor(closrcImage,closrcImage,CV_GRAY2BGR);

	cout << keypoints_1.size() <<endl;
	for(int k=0; k< keypoints_1.size(); k++)
	{
		KeyPoint element;
		
		element = keypoints_1[k];
		circle(closrcImage, element.pt, 2, 
			             Scalar(0,0,255), -1);
	}

	resize(closrcImage,closrcImage,Size(1000,800));
	imshow("final1",closrcImage);
	

	//Filter2, by linethinning

	Mat thinning_image;
	thrImage.copyTo(thinning_image);
	thinning(thinning_image);

	vector <KeyPoint> keypoints_2;
	for(int k=0; k< keypoints_1.size(); k++)
	{
		KeyPoint element;
		element = keypoints_1[k];

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

		Mat kernel = thinning_image(Rect(key_x,key_y,3,3));
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

		if(nwhite <=3)
		{

				keypoints_2.push_back(element);
		}

	}

	Mat closrcImage2; //closrcImage : circle를 위한 복사이미지
	imageROI.copyTo(closrcImage2); 

	cvtColor(closrcImage2,closrcImage2,CV_GRAY2BGR);
	for(int k=0; k< keypoints_2.size(); k++)
	{
		KeyPoint element;
		
		element = keypoints_2[k];
		circle(closrcImage2, element.pt, 2, 
			             Scalar(0,0,255), -1);

	}	

	resize(closrcImage2,closrcImage2,Size(1000,800));
	imshow("final",closrcImage2);

	waitKey();


	return 0;
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat *pMat = (Mat *)param;
	Mat image = Mat(*pMat);
	switch(event)
	{
 	case EVENT_LBUTTONDOWN:
		{
			rectangle(image, Point(x-5, y-5), Point(x+5, y+5), Scalar(255, 0, 0),2);
			cout << x-5 << "," << y-5 <<endl;
		}
		break; 
	}
	imshow("mouseimage", image);
}

void thinningIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

void thinning(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}