#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>

using namespace std;
using namespace cv;

/// ȫ�ֱ���  
Mat src;

//��������
Mat getImage(int source);

int main()
{
	ofstream fout("caliberation_result.txt");  /**    ���涨�������ļ�     **/

	int r = 5;
	int count = 0;
	int image_count = 19;				/****    ͼ������     ****/
	int successImageNum = 0;			/****	�ɹ���ȡ�ǵ������ͼ����	****/

	char imageFileName[50] = "1.jpg";

	Mat image, Extractcorner;
	Mat imageGray;

	vector<Point2f> corners;						 /****    ����ÿ��ͼ���ϼ�⵽�Ľǵ�       ****/
	vector<vector<Point2f>> corners_Seq;			/****  �����⵽�����нǵ�       ****/
	vector<Mat> image_Seq;						   /****    ����ÿ��ͼ��           ****/

	Size board_size = Size(6, 9);					 /****    �������ÿ�С��еĽǵ���       ****/


	for (int n = 0; n < image_count;)
	{
		cout << "Fream #" << n + 1 << "..." << endl;

		image = getImage(0);
		waitKey(30);
		imshow("ԭͼ", image);
		cvtColor(image, imageGray, CV_RGB2GRAY);
		/************************************************************************
		��ȡ���̽ǵ�
		*************************************************************************/
		bool patternfound = findChessboardCorners(imageGray, board_size, corners, CALIB_CB_FAST_CHECK + CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		if (!patternfound)
		{
			cout << "cannot find chessboard corners!\n" << endl;
			imshow("ʧ����Ƭ", image);
			continue;
		}
		else
		{
			/* �����ؾ�ȷ�� */
			cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), cv::TermCriteria(2, 30, 0.001));
			/* ���Ƽ�⵽�Ľǵ㲢���� */
			for (int i = 0; i < corners.size(); i++)
			{
				circle(image, corners[i], r, Scalar(255, 0, 255), 4, 8, 0);
			}
			imshow("�����ؾ�ȷ�ǵ�", image);
			sprintf(imageFileName, "C:\\Users\\QIAO\\Documents\\Visual Studio 2015\\Projects\\�ǵ���\\Test\\picture\\%d.jpg", n + 1);
			imwrite(imageFileName, image);

			count = count + corners.size();
			corners_Seq.push_back(corners);
			image_Seq.push_back(image);

			successImageNum++;
			n++;
		}
	}  //end for()
	cout << "�ǵ���ȡ���!\n" << endl;

	/************************************************************************
	���������
	*************************************************************************/
	cout << "��ʼ���ꡤ����������������\n" << endl;

	Size square_size = Size(20, 20);
	vector<vector<Point3f>> object_Points;				/****  ���涨����Ͻǵ����ά���� �˴�����������ϵ�е��������Ϣ  ****/
	vector<int> point_counts;

	Mat image_counts = Mat(1, count, CV_32FC2, Scalar::all(0));			 /*****   ������ȡ�����нǵ�   *****/

	/* ��ʼ��������Ͻǵ����ά���� */
	for (int t = 0; t < successImageNum; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < board_size.height; i++)
		{
			for (int j = 0; j < board_size.width; j++)
			{
				/* ���趨��������������ϵ��z=0��ƽ���� */
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);      //object_Points��һ����ά���飬ÿһ�д���һ����Ƭ�ϵĽǵ���Ϣ
	}
	for (int i = 0; i < successImageNum; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	/*  ��ʼ����   */
	Size image_size = image_Seq[0].size();
	cv::Matx33d intrinsic_matrix;    /*****    ������ڲ�������    ****/
	cv::Vec4d distortion_coeffs;     /* �������4������ϵ����k1,k2,k3,k4*/
	std::vector<cv::Vec3d> rotation_vectors;                           /* ÿ��ͼ�����ת���� */
	std::vector<cv::Vec3d> translation_vectors;                        /* ÿ��ͼ���ƽ������ */
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 10, 1e-8));
	cout << "������ɣ�\n";

	/************************************************************************
	�Զ�������������
	*************************************************************************/
	cout << "��ʼ���۱궨�����������������������������" << endl;

	double total_err = 0.0;
	double err = 0.0;
	vector<Point2f> image_points2;

	std::cout << "ÿ��ͼ��Ķ�����" << endl;
	std::cout << "ÿ��ͼ��Ķ�����" << endl << endl;

	for (int i = 0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		/****    ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ��     ****/
		fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
		/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
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
		cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
	}

	cout << "����ƽ����" << total_err / image_count << "����" << endl;
	fout << "����ƽ����" << total_err / image_count << "����" << endl << endl;
	std::cout << "������ɣ�" << endl;

	/************************************************************************
	��ʾ������
	*************************************************************************/
#if 1
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);

	cout << "�������ͼ��" << endl;
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
	cout << "�������" << endl;

#endif

	/************************************************************************
	����һ��ͼƬ
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
		//�����ӳ���С,�˵�ϵ��ԽС�ӳ�Խ��
		new_intrinsic_mat.at<double>(0, 0) *= 0.5;
		new_intrinsic_mat.at<double>(1, 1) *= 0.5;
		//����У��ͼ���ģ�һ�㲻���ı�
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


