#include "opencv2/opencv.hpp"
#include <iostream>
#include <time.h>

using namespace cv; //�ŧi opencv �禡�w���R�W�Ŷ�
using namespace std; //�ŧi C++�禡�w���R�W�Ŷ�

/** Global variables */
String face_cascade_name = "data/haarcascade_frontalface_alt.xml"; //�����H�y�r�������V�m�ƾ�
vector<Mat> im_datasets;
CascadeClassifier face_cascade; //�إ��r������������
Rect faceROI; //�H�y���ϰ�
Mat im; //��J�v��
int option = 0; //�w�]�ﶵ
int width, height;

/** Function Headers */
void detectAndDisplay(void); //�����H�y���禡�ŧi

void changeSkinColor(Mat& im, Rect faceROI) {
	Mat hsv; //�s���ഫ��HSV����
	Mat temp_im = im(faceROI) * 2; //�s��չL������ϰ�
	Mat processed_faceROI; //�ݳQ�վ㽧��ϰ쪺mask
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20)); //���ȡB�I�k�����c����
	cvtColor(im(faceROI), hsv, COLOR_BGR2HSV); //�NRGB im���H�y�ϰ��নHSV
	inRange(hsv, Scalar(0, 40, 40), Scalar(50, 180, 240), processed_faceROI); //�佧�⪺�ϰ�
	dilate(processed_faceROI, processed_faceROI, element); //����
	erode(processed_faceROI, processed_faceROI, element); //�I�k
	temp_im.copyTo(im(faceROI), processed_faceROI); //�Q�ο��ȤΫI�k�L�Უ�ͪ�Mask�N�վ�L�᪺����ϰ�s���t�@�i��
}

void corner_detection(Mat& im, Rect faceROI) {
	Mat im_gray; //�s������Ƕ�
	Mat dst_im_gray; //�s��B�z�L�᪺�Ƕ���
	Mat dst_norm, dst_norm_scaled; 
	int threshold = 150; //���I�����һݪ����e��
	cvtColor(im(faceROI), im_gray, COLOR_BGR2GRAY); //�����Ƕ�
	cornerHarris(im_gray, dst_im_gray, 3, 3, 0.04); //�z�LHarris�������I(https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345)
	normalize(dst_im_gray, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); //�N�g�Lharris�������Ƕ��Ϫ��C��pixel�����W��
	convertScaleAbs(dst_norm, dst_norm_scaled); //�N�C��pixel�ܦ�����
	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > threshold)
			{
				circle(im(faceROI), Point(j, i), 5, Scalar(0), 2, 8, 1); //�e���(�v��, �y��. �骺�b�|, �C��, �u���ʲ�, ��骺����, �b�|�Ȫ��p���I���)
			}
		}
	}
}

void change_face(Mat& im, Rect faceROI, vector<Mat> im_datasets) {
	int x = rand() % 2; //rand() % x + y, �d���ܱqy�}�l��x�Ӽ�
	vector<Rect> faces; //�إߤH�yROI �V�q
	Rect faceROI2; //�ݶK�W���H�y�ϰ�
	Mat dst_img; //�s���Y�p���P��ϤH�y�ϰ�@�ˤj���ݶK�W���H�y�ϰ�
	Mat im_gray; //�Ƕ��v������

	cvtColor(im_datasets[1], im_gray, COLOR_BGR2GRAY); //�m��v����Ƕ�
	equalizeHist(im_gray, im_gray); //�Ƕ��Ȥ�ϵ���(���۰ʼW�j)�C�Y���T�~��n�A�i����

	//�H�y�r������
	face_cascade.detectMultiScale(im_gray, faces, 1.1, 4, 0, Size(80, 80));

	//��o�̤j�H�y�� ROI�ƾ�
	if (faces.size() > 0) {
		int largest_area = -999;
		int largest_i;
		for (int i = 0; i < faces.size(); i++) //�ΰj��Ū���Ҧ��H�y ROI
		{
			//�w�q�v������ ROI
			if (largest_area < faces[i].height)
			{
				largest_area = faces[i].height;
				largest_i = i;
			}
		}

		faceROI2 = faces[largest_i]; //�T�w�ݶK�W���H�y�ϰ�

		resize(im_datasets[x](faceROI2), dst_img, Size(im(faceROI).cols, im(faceROI).rows), INTER_LINEAR); //�N�ݶK�W���H�y�����ܱo���Ϫ��H�y�����@�ˤj
		// �y����jROI�A�Ϸs�y��������л\���T�����H�y(�i����)
		int d = 25;
		faceROI2.x = faceROI2.x - d;
		faceROI2.y = faceROI2.y - d;
		faceROI2.width = faceROI2.width + 2 * d;
		faceROI2.height = faceROI2.height + 2 * d;

		//�קKfaceROI�W�X�e��
		if (faceROI2.x > im_datasets[x].cols) { //�P�_�_�l�y��x�O�_�W�L�Ϥ���width
			faceROI2.x = im_datasets[x].cols;
		}
		else if (faceROI2.x + faceROI2.width > im_datasets[x].cols) {
			faceROI2.width = faceROI2.width - (faceROI2.x + faceROI2.width - im_datasets[x].cols); //�Nwidth��h�W�L��ɪ��ƭ�
		}
		if (faceROI2.y > im_datasets[x].rows) { //�P�_�_�l�y��y�O�_�W�L�Ϥ���height
			faceROI2.y = im_datasets[x].rows;
		}
		else if (faceROI2.y + faceROI2.height > im_datasets[x].rows) {
			faceROI2.height = faceROI2.height - (faceROI2.y + faceROI2.height - im_datasets[x].rows); //�Nheight��h�W�L��ɪ��ƭ�
		}
	}
	dst_img.copyTo(im(faceROI));
}

//�w�q�ƹ������禡 mouse_callback
static void mouse_callback(int event, int x, int y, int flags, void *)
{
	// ��ƹ����U����A�ھ��I�諸(x,y)��m�A�o��ﶵ (option) �ƭ�
	switch (event) {
	case EVENT_LBUTTONDOWN: //������U���ƥ�
		if ((x >= 29 && x <= 209) && (y >= 400 && y <= 470)) {
			option = 1;
		}
		else if ((x >= 229 && x <= 409) && (y >= 400 && y <= 470)) {
			option = 2;
		}
		else if ((x >= 429 && x <= 609) && (y >= 400 && y <= 470)) {
			option = 3;
		}
		else
			option = 0;
		break;
	}
}

int main(void)
{
	VideoCapture cap("data/sleepy.mp4"); //Ū���v���ά۾�
	//VideoCapture cap(0); //�Y�u���@���ṳ�Y�h�]0�A�Y���h���ṳ�Y�A�h�n�]>0(���]�h�|���q�{���ṳ�Y)

	stringstream img_number; 

	for (int i = 0; i < 2; i++) {
		img_number.clear();
		img_number.str("");
		img_number << i; //�Nint�ǤJstring�������ܼ�
		Mat img = imread(img_number.str() + ".png"); //Ū����Ʈw�����Ϥ�
		im_datasets.push_back(img); //�N�Ϥ��s�Jvecter��
	}

	if (!cap.isOpened()) return 0; //����Ū���T���B�z

	//�פJ�H�y�r�������V�m�ƾ�
	if (!face_cascade.load(face_cascade_name)) 
	{ printf("--(!)Error loading face cascade\n");
	  waitKey(0); //wait for key(���ݥΤ���U�S�w����Ӹ��X�j��) ���N��,0���ܪ�ܵL������
	  return -1; 
	};

	while (char(waitKey(1)) != 27 && cap.isOpened()) //����L�S�� Esc�A�H�ε��T���󦨥\�}�ҮɡA������� while �j��
	{
		cap >> im; //������T���e��
		if (im.empty()) //�p�G�S��� �e��
		{
			printf(" --(!) No captured im -- Break!");  //��ܿ��~�T��
			break;
		}
		//�w�q�����W�� namedWindow
		namedWindow("window");
		rectangle(im, Rect(29, 400, 180, 70), Scalar(150, 150, 0), 2, 2, 0); //option1����l (29,400) ~ (209,470)
		rectangle(im, Rect(229, 400, 180, 70), Scalar(150, 150, 0), 2, 2, 0); //option2����l (229,400) ~ (409,470)
		rectangle(im, Rect(429, 400, 180, 70), Scalar(150, 150, 0), 2, 2, 0); //option3����l (429,400) ~ (609,470)
		putText(im, "change skin", Point(40, 425), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option1���e
		putText(im, "color", Point(84, 455), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option1���e
		putText(im, "corner", Point(280, 425), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option2���e
		putText(im, "detection", Point(260, 455), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option2���e
		putText(im, "change face", Point(440, 440), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option3���e
		imshow("window", im); //im.rows * im.cols = 480 * 640

		/*�����H�y�A�����AR�Ϲ��ĦX���G*/
		detectAndDisplay();
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(void){
	/*�H�y��������*/
	vector<Rect> faces; //�إߤH�yROI �V�q
	Mat im_gray; //�Ƕ��v������

	cvtColor(im, im_gray, COLOR_BGR2GRAY); //�m��v����Ƕ�
	equalizeHist(im_gray, im_gray); //�Ƕ��Ȥ�ϵ���(���۰ʼW�j)�C�Y���T�~��n�A�i����

	//�H�y�r������
	face_cascade.detectMultiScale(im_gray, faces, 1.1, 4, 0, Size(80, 80));

	//��o�̤j�H�y�� ROI�ƾ�
	if (faces.size() > 0) {
		int largest_area = -999;
		int largest_i;
		for (int i = 0; i < faces.size(); i++) //�ΰj��Ū���Ҧ��H�y ROI
		{
			//�w�q�v������ ROI
			if (largest_area < faces[i].height)
			{
				largest_area = faces[i].height;
				largest_i = i;
			}
		}

		faceROI = faces[largest_i]; //�N�̤j�H�y�� ROI�ƾڦs�J faceROI

		// �y����jROI�A�Ϸs�y��������л\���T�����H�y(�i����)
		int d = 25;
		faceROI.x = faceROI.x - d;
		faceROI.y = faceROI.y - d;
		faceROI.width = faceROI.width + 2 * d;
		faceROI.height = faceROI.height + 2 * d;

		//�קKfaceROI�W�X�e��
		if (faceROI.x > im.cols) { //�P�_�_�l�y��x�O�_�W�L�Ϥ���width
			faceROI.x = im.cols; 
		}
		else if (faceROI.x + faceROI.width > im.cols) {
			faceROI.width = faceROI.width - (faceROI.x + faceROI.width - im.cols); //�Nwidth��h�W�L��ɪ��ƭ�
		}
		if (faceROI.y > im.rows) { //�P�_�_�l�y��y�O�_�W�L�Ϥ���height
			faceROI.y = im.rows; 
		}
		else if (faceROI.y + faceROI.height > im.rows) {
			faceROI.height = faceROI.height - (faceROI.y + faceROI.height - im.rows); //�Nheight��h�W�L��ɪ��ƭ�
		}
		//ø�s�H�y�ϰ�x�ή�
		rectangle(im, Rect(faceROI.x, faceROI.y, faceROI.width, faceROI.height), Scalar(0, 0, 255), 2, 2, 0); //(�v��,�y��(���W��x�y��,���W��y�y��,��,�e), �u���C��, �ʫ�, �u������, �첾)
		//�b�H�y�ϰ�x�ήؤW��g�Ǹ�
		putText(im, "M10815068", Point(faceROI.x, faceROI.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2, 0); //(�v��,�n��W�����e,���e�����W���y��,�r���榡,�r���j�p,�C��,�r���ʫ�,�u������)
		//�]�w�ƹ������禡 setMouseCallback
		setMouseCallback("window", mouse_callback);
		//�ھ� option �ﶵ�A�B�zROI�v��
		switch (option) {
		case 1:
			changeSkinColor(im, faceROI);
			break;
		case 2:
			corner_detection(im, faceROI);
			break;
		case 3:
			change_face(im, faceROI, im_datasets);
		}
	}
	//��ܼv��
	imshow("window", im);
	}