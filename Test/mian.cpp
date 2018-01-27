#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>

using namespace std;
using namespace cv;

/// 全局变量  
Mat src;

//函数声明
Mat getImage(int source);

int main()
{
	ofstream fout("caliberation_result.txt");  /**    保存定标结果的文件     **/

	int r = 5;
	int count = 0;
	int image_count = 19;				/****    图像数量     ****/
	int successImageNum = 0;			/****	成功提取角点的棋盘图数量	****/

	char imageFileName[50] = "1.jpg";

	Mat image, Extractcorner;
	Mat imageGray;

	vector<Point2f> corners;						 /****    缓存每幅图像上检测到的角点       ****/
	vector<vector<Point2f>> corners_Seq;			/****  保存检测到的所有角点       ****/
	vector<Mat> image_Seq;						   /****    缓存每幅图像           ****/

	Size board_size = Size(6, 9);					 /****    定标板上每行、列的角点数       ****/


	for (int n = 0; n < image_count;)
	{
		cout << "Fream #" << n + 1 << "..." << endl;

		image = getImage(0);
		waitKey(30);
		imshow("原图", image);
		cvtColor(image, imageGray, CV_RGB2GRAY);
		/************************************************************************
		提取棋盘角点
		*************************************************************************/
		bool patternfound = findChessboardCorners(imageGray, board_size, corners, CALIB_CB_FAST_CHECK + CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		if (!patternfound)
		{
			cout << "cannot find chessboard corners!\n" << endl;
			imshow("失败照片", image);
			continue;
		}
		else
		{
			/* 亚像素精确化 */
			cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), cv::TermCriteria(2, 30, 0.001));
			/* 绘制检测到的角点并保存 */
			for (int i = 0; i < corners.size(); i++)
			{
				circle(image, corners[i], r, Scalar(255, 0, 255), 4, 8, 0);
			}
			imshow("亚像素精确角点", image);
			sprintf(imageFileName, "C:\\Users\\QIAO\\Documents\\Visual Studio 2015\\Projects\\角点检测\\Test\\picture\\%d.jpg", n + 1);
			imwrite(imageFileName, image);

			count = count + corners.size();
			corners_Seq.push_back(corners);
			image_Seq.push_back(image);

			successImageNum++;
			n++;
		}
	}  //end for()
	cout << "角点提取完毕!\n" << endl;

	/************************************************************************
	摄像机定标
	*************************************************************************/
	cout << "开始定标·········\n" << endl;

	Size square_size = Size(20, 20);
	vector<vector<Point3f>> object_Points;				/****  保存定标板上角点的三维坐标 此处是世界坐标系中的坐标点信息  ****/
	vector<int> point_counts;

	Mat image_counts = Mat(1, count, CV_32FC2, Scalar::all(0));			 /*****   保存提取的所有角点   *****/

	/* 初始化定标板上角点的三维坐标 */
	for (int t = 0; t < successImageNum; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < board_size.height; i++)
		{
			for (int j = 0; j < board_size.width; j++)
			{
				/* 假设定标板放在世界坐标系中z=0的平面上 */
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);      //object_Points是一个二维数组，每一行储存一张照片上的角点信息
	}
	for (int i = 0; i < successImageNum; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	/*  开始定标   */
	Size image_size = image_Seq[0].size();
	cv::Matx33d intrinsic_matrix;    /*****    摄像机内参数矩阵    ****/
	cv::Vec4d distortion_coeffs;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
	std::vector<cv::Vec3d> rotation_vectors;                           /* 每幅图像的旋转向量 */
	std::vector<cv::Vec3d> translation_vectors;                        /* 每幅图像的平移向量 */
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 10, 1e-8));
	cout << "定标完成！\n";

	/************************************************************************
	对定标结果进行评价
	*************************************************************************/
	cout << "开始评价标定结果·············" << endl;

	double total_err = 0.0;
	double err = 0.0;
	vector<Point2f> image_points2;

	std::cout << "每幅图像的定标误差：" << endl;
	std::cout << "每幅图像的定标误差：" << endl << endl;

	for (int i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		/****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
		fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = corners_Seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (size_t i = 0; i != tempImagePoint.size(); i++)
		{
			image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}

	cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
	std::cout << "评价完成！" << endl;

	/************************************************************************
	显示定标结果
	*************************************************************************/
#if 1
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);

	cout << "保存矫正图像" << endl;
	for (int i = 0; i != image_count; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
		Mat t = image_Seq[i].clone();
		cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR);
		string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += "_d.jpg";
		imwrite(imageFileName, t);
	}
	cout << "保存结束" << endl;

#endif

	/************************************************************************
	测试一张图片
	*************************************************************************/
#if 1
	cout << "TestImage ..." << endl;
	vector<Point>a;
	vector<Point>b;
	while (1)
	{
		Mat distort_img;
		Mat undistort_img;
		Mat intrinsic_mat(intrinsic_matrix), new_intrinsic_mat;
		distort_img = imread("testimage.jpg", 1);
		intrinsic_mat.copyTo(new_intrinsic_mat);
		//调节视场大小,乘的系数越小视场越大
		new_intrinsic_mat.at<double>(0, 0) *= 0.5;
		new_intrinsic_mat.at<double>(1, 1) *= 0.5;
		//调节校正图中心，一般不做改变
		new_intrinsic_mat.at<double>(0, 2) += 0;
		new_intrinsic_mat.at<double>(1, 2) += 0;

		//fisheye::undistortImage(a, b, intrinsic_matrix, distortion_coeffs, new_intrinsic_mat);
		fisheye::undistortImage(distort_img, undistort_img, intrinsic_matrix, distortion_coeffs, new_intrinsic_mat);
		imshow("output.jpg", undistort_img);
		waitKey(0);

	}

#endif


	waitKey(0);
	return 0;

}

Mat getImage(int source)
{
	Mat tempImage;
	if (source == 0)
	{
		VideoCapture cap(0);
		if (!cap.isOpened())
		{
			cout << "cannot open camera!\n" << endl;
		}
		cap >> tempImage;
		return tempImage;
	}
	else
	{
		tempImage = imread("chess.jpg", 1);
		if (!tempImage.data)
		{
			cout << "there is no picture in the folder.\n" << endl;
		}
		return tempImage;
	}
}


