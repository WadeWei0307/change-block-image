#include "opencv2/opencv.hpp"
#include <iostream>
#include <time.h>

using namespace cv; //宣告 opencv 函式庫的命名空間
using namespace std; //宣告 C++函式庫的命名空間

/** Global variables */
String face_cascade_name = "data/haarcascade_frontalface_alt.xml"; //正面人臉瀑布偵測訓練數據
vector<Mat> im_datasets;
CascadeClassifier face_cascade; //建立瀑布分類器物件
Rect faceROI; //人臉的區域
Mat im; //輸入影像
int option = 0; //預設選項
int width, height;

/** Function Headers */
void detectAndDisplay(void); //偵測人臉的函式宣告

void changeSkinColor(Mat& im, Rect faceROI) {
	Mat hsv; //存放轉換成HSV的圖
	Mat temp_im = im(faceROI) * 2; //存放調過的膚色區域
	Mat processed_faceROI; //需被調整膚色區域的mask
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20)); //膨脹、侵蝕的結構元素
	cvtColor(im(faceROI), hsv, COLOR_BGR2HSV); //將RGB im的人臉區域轉成HSV
	inRange(hsv, Scalar(0, 40, 40), Scalar(50, 180, 240), processed_faceROI); //找膚色的區域
	dilate(processed_faceROI, processed_faceROI, element); //膨脹
	erode(processed_faceROI, processed_faceROI, element); //侵蝕
	temp_im.copyTo(im(faceROI), processed_faceROI); //利用膨脹及侵蝕過後產生的Mask將調整過後的膚色區域存放到另一張圖
}

void corner_detection(Mat& im, Rect faceROI) {
	Mat im_gray; //存放原圖轉灰階
	Mat dst_im_gray; //存放處理過後的灰階圖
	Mat dst_norm, dst_norm_scaled; 
	int threshold = 150; //角點偵測所需的門檻值
	cvtColor(im(faceROI), im_gray, COLOR_BGR2GRAY); //原圖轉灰階
	cornerHarris(im_gray, dst_im_gray, 3, 3, 0.04); //透過Harris偵測角點(https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345)
	normalize(dst_im_gray, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); //將經過harris偵測的灰階圖的每個pixel做正規化
	convertScaleAbs(dst_norm, dst_norm_scaled); //將每個pixel變成正數
	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > threshold)
			{
				circle(im(faceROI), Point(j, i), 5, Scalar(0), 2, 8, 1); //畫圓圈(影像, 座標. 圈的半徑, 顏色, 線條粗細, 圓圈的類型, 半徑值的小數點位數)
			}
		}
	}
}

void change_face(Mat& im, Rect faceROI, vector<Mat> im_datasets) {
	int x = rand() % 2; //rand() % x + y, 範圍表示從y開始取x個數
	vector<Rect> faces; //建立人臉ROI 向量
	Rect faceROI2; //待貼上的人臉區域
	Mat dst_img; //存放縮小成與原圖人臉區域一樣大的待貼上的人臉區域
	Mat im_gray; //灰階影像物件

	cvtColor(im_datasets[1], im_gray, COLOR_BGR2GRAY); //彩色影像轉灰階
	equalizeHist(im_gray, im_gray); //灰階值方圖等化(對比自動增強)。若視訊品質好，可不用

	//人臉瀑布偵測
	face_cascade.detectMultiScale(im_gray, faces, 1.1, 4, 0, Size(80, 80));

	//獲得最大人臉的 ROI數據
	if (faces.size() > 0) {
		int largest_area = -999;
		int largest_i;
		for (int i = 0; i < faces.size(); i++) //用迴圈讀取所有人臉 ROI
		{
			//定義影像中的 ROI
			if (largest_area < faces[i].height)
			{
				largest_area = faces[i].height;
				largest_i = i;
			}
		}

		faceROI2 = faces[largest_i]; //確定待貼上的人臉區域

		resize(im_datasets[x](faceROI2), dst_img, Size(im(faceROI).cols, im(faceROI).rows), INTER_LINEAR); //將待貼上的人臉部分變得跟原圖的人臉部分一樣大
		// 稍為放大ROI，使新臉能夠完整覆蓋視訊中的人臉(可不做)
		int d = 25;
		faceROI2.x = faceROI2.x - d;
		faceROI2.y = faceROI2.y - d;
		faceROI2.width = faceROI2.width + 2 * d;
		faceROI2.height = faceROI2.height + 2 * d;

		//避免faceROI超出畫面
		if (faceROI2.x > im_datasets[x].cols) { //判斷起始座標x是否超過圖片的width
			faceROI2.x = im_datasets[x].cols;
		}
		else if (faceROI2.x + faceROI2.width > im_datasets[x].cols) {
			faceROI2.width = faceROI2.width - (faceROI2.x + faceROI2.width - im_datasets[x].cols); //將width減去超過邊界的數值
		}
		if (faceROI2.y > im_datasets[x].rows) { //判斷起始座標y是否超過圖片的height
			faceROI2.y = im_datasets[x].rows;
		}
		else if (faceROI2.y + faceROI2.height > im_datasets[x].rows) {
			faceROI2.height = faceROI2.height - (faceROI2.y + faceROI2.height - im_datasets[x].rows); //將height減去超過邊界的數值
		}
	}
	dst_img.copyTo(im(faceROI));
}

//定義滑鼠反應函式 mouse_callback
static void mouse_callback(int event, int x, int y, int flags, void *)
{
	// 當滑鼠按下左鍵，根據點選的(x,y)位置，得到選項 (option) 數值
	switch (event) {
	case EVENT_LBUTTONDOWN: //左鍵按下的事件
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
	VideoCapture cap("data/sleepy.mp4"); //讀取影片或相機
	//VideoCapture cap(0); //若只有一個攝像頭則設0，若有多個攝像頭，則要設>0(不設則會選默認的攝像頭)

	stringstream img_number; 

	for (int i = 0; i < 2; i++) {
		img_number.clear();
		img_number.str("");
		img_number << i; //將int傳入string類型的變數
		Mat img = imread(img_number.str() + ".png"); //讀取資料庫中的圖片
		im_datasets.push_back(img); //將圖片存入vecter裡
	}

	if (!cap.isOpened()) return 0; //不能讀視訊的處理

	//匯入人臉瀑布偵測訓練數據
	if (!face_cascade.load(face_cascade_name)) 
	{ printf("--(!)Error loading face cascade\n");
	  waitKey(0); //wait for key(等待用戶按下特定按鍵來跳出迴圈) 的意思,0的話表示無限等待
	  return -1; 
	};

	while (char(waitKey(1)) != 27 && cap.isOpened()) //當鍵盤沒按 Esc，以及視訊物件成功開啟時，持續執行 while 迴圈
	{
		cap >> im; //抓取視訊的畫面
		if (im.empty()) //如果沒抓到 畫面
		{
			printf(" --(!) No captured im -- Break!");  //顯示錯誤訊息
			break;
		}
		//定義視窗名稱 namedWindow
		namedWindow("window");
		rectangle(im, Rect(29, 400, 180, 70), Scalar(150, 150, 0), 2, 2, 0); //option1的格子 (29,400) ~ (209,470)
		rectangle(im, Rect(229, 400, 180, 70), Scalar(150, 150, 0), 2, 2, 0); //option2的格子 (229,400) ~ (409,470)
		rectangle(im, Rect(429, 400, 180, 70), Scalar(150, 150, 0), 2, 2, 0); //option3的格子 (429,400) ~ (609,470)
		putText(im, "change skin", Point(40, 425), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option1內容
		putText(im, "color", Point(84, 455), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option1內容
		putText(im, "corner", Point(280, 425), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option2內容
		putText(im, "detection", Point(260, 455), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option2內容
		putText(im, "change face", Point(440, 440), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(250, 100, 150), 2, 0); //option3內容
		imshow("window", im); //im.rows * im.cols = 480 * 640

		/*偵測人臉，並顯示AR圖像融合結果*/
		detectAndDisplay();
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(void){
	/*人臉偵測部分*/
	vector<Rect> faces; //建立人臉ROI 向量
	Mat im_gray; //灰階影像物件

	cvtColor(im, im_gray, COLOR_BGR2GRAY); //彩色影像轉灰階
	equalizeHist(im_gray, im_gray); //灰階值方圖等化(對比自動增強)。若視訊品質好，可不用

	//人臉瀑布偵測
	face_cascade.detectMultiScale(im_gray, faces, 1.1, 4, 0, Size(80, 80));

	//獲得最大人臉的 ROI數據
	if (faces.size() > 0) {
		int largest_area = -999;
		int largest_i;
		for (int i = 0; i < faces.size(); i++) //用迴圈讀取所有人臉 ROI
		{
			//定義影像中的 ROI
			if (largest_area < faces[i].height)
			{
				largest_area = faces[i].height;
				largest_i = i;
			}
		}

		faceROI = faces[largest_i]; //將最大人臉的 ROI數據存入 faceROI

		// 稍為放大ROI，使新臉能夠完整覆蓋視訊中的人臉(可不做)
		int d = 25;
		faceROI.x = faceROI.x - d;
		faceROI.y = faceROI.y - d;
		faceROI.width = faceROI.width + 2 * d;
		faceROI.height = faceROI.height + 2 * d;

		//避免faceROI超出畫面
		if (faceROI.x > im.cols) { //判斷起始座標x是否超過圖片的width
			faceROI.x = im.cols; 
		}
		else if (faceROI.x + faceROI.width > im.cols) {
			faceROI.width = faceROI.width - (faceROI.x + faceROI.width - im.cols); //將width減去超過邊界的數值
		}
		if (faceROI.y > im.rows) { //判斷起始座標y是否超過圖片的height
			faceROI.y = im.rows; 
		}
		else if (faceROI.y + faceROI.height > im.rows) {
			faceROI.height = faceROI.height - (faceROI.y + faceROI.height - im.rows); //將height減去超過邊界的數值
		}
		//繪製人臉區域矩形框
		rectangle(im, Rect(faceROI.x, faceROI.y, faceROI.width, faceROI.height), Scalar(0, 0, 255), 2, 2, 0); //(影像,座標(左上方x座標,左上方y座標,長,寬), 線條顏色, 粗度, 線條類型, 位移)
		//在人臉區域矩形框上方寫學號
		putText(im, "M10815068", Point(faceROI.x, faceROI.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2, 0); //(影像,要放上的內容,內容的左上角座標,字的格式,字的大小,顏色,字的粗度,線條類型)
		//設定滑鼠反應函式 setMouseCallback
		setMouseCallback("window", mouse_callback);
		//根據 option 選項，處理ROI影像
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
	//顯示影像
	imshow("window", im);
	}