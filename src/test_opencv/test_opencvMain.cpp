#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>


using namespace cv;
#define ROW_SAMPLE (100)

#if 0
void test_knn()
{
	Mat img = imread("E:/opencv/opencv/sources/samples/data/digits.png");
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	int b = 20;
	int m = gray.rows / b;   //原图为1000*2000
	int n = gray.cols / b;   //裁剪为5000个20*20的小图块
	Mat data,labels;   //特征矩阵
	for (int i = 0; i < n; i++)
	{
		int offsetCol = i*b; //列上的偏移量
		for (int j = 0; j < m; j++)
		{
			int offsetRow = j*b;  //行上的偏移量
			//截取20*20的小块
			Mat tmp;
			gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			data.push_back(tmp.reshape(0,1));  //序列化后放入特征矩阵
			labels.push_back((int)j / 5);  //对应的标注
		}

	}
	data.convertTo(data, CV_32F); //uchar型转换为cv_32f
	int samplesNum = data.rows;
	int trainNum = 3000;
	Mat trainData, trainLabels;
	trainData = data(Range(0, trainNum), Range::all());   //前3000个样本为训练数据
	trainLabels = labels(Range(0, trainNum), Range::all());

	//使用KNN算法
	int K = 5;
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	Ptr<KNearest> model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tData);

	//预测分类
	double train_hr = 0, test_hr = 0;
	Mat response;
	// compute prediction error on train and test data
	for (int i = 0; i < samplesNum; i++)
	{
		Mat sample = data.row(i);
		float r = model->predict(sample);   //对所有行进行预测
		//预测结果与原结果相比，相等为1，不等为0
		r = std::abs(r - labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;          

		if (i < trainNum)
			train_hr += r;  //累积正确数
		else
			test_hr += r;
	}

	test_hr /= samplesNum - trainNum;
	train_hr = trainNum > 0 ? train_hr / trainNum : 1.;

	printf("accuracy: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);
	waitKey(0);
	

	::system("pause");
}


void test_getpolar(){


	std::string image_name="model_src_img.jpg";
	cv::Mat image;
	image=cv::imread(image_name,CV_LOAD_IMAGE_COLOR);
	cv::Mat image1=image.clone();
	if(!image.data)
	{
		printf("no image data!\n");
		::system("pause");
	}
	std::vector<cv::Point2f> point_list;
	point_list.push_back(cv::Point2f(680,547));
	point_list.push_back(cv::Point2f(713,495));
	point_list.push_back(cv::Point2f(757,453));
	point_list.push_back(cv::Point2f(873,416));
	point_list.push_back(cv::Point2f(987,448));
	point_list.push_back(cv::Point2f(1069,542));

	cv::RotatedRect rotate_rect=FOpenCV::GetEllipse(point_list);
	cv::Point2f center=rotate_rect.center;
	FOpenCV::DrawEllipse(image,rotate_rect);
	FOpenCV::DrawLine(image,center,cv::Point2f(680,547));
	FOpenCV::DrawLine(image,center,cv::Point2f(1069,542));
	FOpenCV::DrawPoint(image,center);


	cv::Mat polar_image;
	double polar_d1;
	double polar_d2;
	double dis_pointer_start_ellipse;
	double dis_pointer_end_ellipse;
	bool ret=FOpenCV::GetPolar(image1,rotate_rect,cv::Point2f(864,604),2,cv::Point2f(770,439),1,polar_image,polar_d1,polar_d2,dis_pointer_start_ellipse,dis_pointer_end_ellipse);

	cv::imshow("test opencv",image);
	cv::imshow("polar image",polar_image);
	cv::waitKey(-1);
	return 0;

}
#endif






void test_readrawimage(){


	FILE *fp = fopen("1.raw","rb");
	


	unsigned char *buffer=new unsigned char[522240];
	if(fp!=NULL){
		std::cout<<"open success"<<std::endl;
		int ret=fread(buffer, 1, 522240, fp);

		cv::Mat image=cv::Mat(272,640,CV_8UC3);
		memcpy(image.data,buffer,522240);


		cv::imshow("image",image);
		cv::waitKey(-1);

		fclose(fp);
	}
	else{

		std::cout<<"open failure"<<std::endl;

	}


	::system("pause");
	
	delete [] buffer;

}

void test_yuv2rgb(){

	cv::Mat yuvimg;
	FILE *fp = fopen("1.yuv","rb");
	unsigned char *buffer=new unsigned char[1920*1080*3/2];
	int ret=fread(buffer, 1, 1920*1080*3/2, fp);
	yuvimg.create(1080 * 3/2, 1920, CV_8UC1);
	memcpy(yuvimg.data, buffer, 1920*1080*3/2);

	cv::Mat rgbimg=cv::Mat(1080,1920,CV_8UC3);
	cv::cvtColor(yuvimg, rgbimg, CV_YUV2BGR_I420);
	delete [] buffer;
	buffer=NULL;
	fclose(fp);
	fp=NULL;
	cv::imshow("image",rgbimg);
	cv::waitKey(-1);
}


void test_jpg2bbggrr(int argc,char **argv){
	printf("usage:jpg2bbggrr x.jpg x.bgr width height\n");
	if(argc<5){
		printf("too few parameters \n");
		return;
	}
	printf("argv:%s,%s,%s,%s,%s\n",argv[0],argv[1],argv[2],argv[3],argv[5]);

	const char *filename = argv[1];
	const char *outname = argv[2];
	int width=atoi(argv[3]);
	int height=atoi(argv[4]);
	printf("size:%d,%d\n",width,height);
	if(width<0){
		printf("width error\n");
		return;
	}

	if(height<0){
		printf("height error\n");
		return;
	}

	cv::Mat img = cv::imread(filename);
	if (!img.data)
	{
		printf("read image error\n");
		return;
	}

	//缩放
	cv::resize(img, img, cv::Size(width, height));  //224x224
	

	unsigned char *data = (unsigned char*)img.data;
	int step = img.step;
	printf("step: %d, height: %d, width: %d\n",step, img.rows, img.cols);


	FILE *fp = fopen(outname, "wb");
	if(fp!=NULL){

		int h = img.rows;
		int w = img.cols;
		int c = img.channels();

		for (int k = 0; k<c; k++) {
			for (int i = 0; i<h; i++) {
				for (int j = 0; j<w; j++) {
					fwrite(&data[i*step + j*c + k], sizeof(unsigned char), 1, fp);
				}
			}
		}
		fclose(fp);
		fp=NULL;

	}
	else{
		printf("file open error\n");
	}



}

int main(int argc,char **argv)
{

	//test_detectface();
	//test_readrawimage();
	//test_yuv2rgb();
	test_jpg2bbggrr(argc,argv);
	return 0;


}


