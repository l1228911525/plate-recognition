#include "charsegment.hpp"

using namespace std;
using namespace cv;

void charsegment::PlateCharSegmentMat(const Mat & inputImg,vector <Mat>& dst_mat, Color r)
{
    Mat src_threshold, img_threshold;
    if(inputImg.channels() == 3)
        cvtColor(inputImg, src_threshold, CV_RGB2GRAY);
	//threshold(inputImg ,img_threshold , 120,255 ,CV_THRESH_BINARY );

	if (r == BLUE)
    {
          cv::threshold(src_threshold, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    } else if (r == YELLOW)
    {
          cv::threshold(src_threshold, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
    }



    clearLiuDing(img_threshold);
    Rect outRect;

    ///imshow("ddd", img_threshold);
    ///imshow("sss", ClearBorderMat);
    ///waitKey(0);


	img_threshold.copyTo(src_threshold);
	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4) );
    morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);
    imshow("ddd", img_threshold);
    waitKey(0);
	Mat img_contours;
	img_threshold.copyTo(img_contours);

	vector < vector <Point> > contours;
	findContours(img_contours ,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

	vector< vector <Point> > ::iterator itc = contours.begin();
	vector<RotatedRect> char_rects;

    while( itc != contours.end())
	{
		RotatedRect minArea = minAreaRect(Mat( *itc )); //返回每个轮廓的最小有界矩形区域

		Point2f vertices[4];
		minArea.points(vertices);

  		if(!char_verifySizes(minArea))  //判断矩形轮廓是否符合要求
		{
			itc = contours.erase(itc);
		}
		else
		{
			++itc;
			char_rects.push_back(minArea);

		}
	}
	char_sort(char_rects); //对字符排序

	vector <Mat> char_mat;

	for (int i = 0; i<char_rects.size() ;++i )
	{
		char_mat.push_back( Mat(src_threshold,char_rects[i].boundingRect())) ;
	}

	Mat train_mat(2,3,CV_32FC1);
	int length ;
	dst_mat.resize(7);
	Point2f srcTri[3];
	Point2f dstTri[3];

	for (int i = 0; i< char_mat.size();++i)
	{
		srcTri[0] = Point2f( 0,0 );
		srcTri[1] = Point2f( char_mat[i].cols - 1, 0 );
		srcTri[2] = Point2f( 0, char_mat[i].rows - 1 );
		length = char_mat[i].rows > char_mat[i].cols?char_mat[i].rows:char_mat[i].cols;
		dstTri[0] = Point2f( 0.0, 0.0 );
		dstTri[1] = Point2f( length, 0.0 );
		dstTri[2] = Point2f( 0.0, length );
		train_mat = getAffineTransform( srcTri, dstTri );
		dst_mat[i]=Mat::zeros(length,length,char_mat[i].type());
		warpAffine(char_mat[i],dst_mat[i],train_mat,dst_mat[i].size(),INTER_LINEAR,BORDER_CONSTANT,Scalar(0));
		resize(dst_mat[i],dst_mat[i],Size(20,20));  //尺寸调整为20*20

	}
}

void charsegment::PlateCharSegmentPlate(plate &PlateIn, vector<Mat> &PlateChar)
{
    Mat inputImg = PlateIn.GetPlateMat();
    PlateCharSegmentMat(inputImg, PlateChar, PlateIn.GetPlateColor());
}

bool charsegment::char_verifySizes(const RotatedRect & candidate)
{
	float aspect = 33.0f/20.0f;
	float charAspect = (float) candidate.size.width/ (float)candidate.size.height; //宽高比
	float error = 0.35;
	float minHeight = 11;  //最小高度11
	float maxHeight = 33;  //最大高度33

	float minAspect = 0.20;  //考虑到数字1，最小长宽比为0.15
	float maxAspect = aspect + aspect*error;

	if( charAspect > minAspect && charAspect < maxAspect
		&& candidate.size.height >= minHeight && candidate.size.width< maxHeight) //非0像素点数、长宽比、高度需满足条件
		return true;
	else
		return false;
}

void charsegment::char_sort(vector <RotatedRect > & in_char ) //对字符区域进行排序
{
	vector <RotatedRect >  out_char;
	const int length = 7;           //7个字符
	int index[length] = {0,1,2,3,4,5,6};
	float centerX[length];
	for (int i=0;i < length ; ++ i)
	{
		centerX[i] = in_char[i].center.x;
	}

	for (int j=0;j <length;j++) {
		for (int i=length-2;i >= j;i--)
			if (centerX[i] > centerX[i+1])
			{
				float t=centerX[i];
				centerX[i]=centerX[i+1];
				centerX[i+1]=t;

				int tt = index[i];
				index[i] = index[i+1];
				index[i+1] = tt;
			}
	}

	for(int i=0;i<length ;i++)
		out_char.push_back(in_char[(index[i])]);

	in_char.clear();     //清空in_char
	in_char = out_char; //将排序好的字符区域向量重新赋值给in_char
}


bool charsegment::clearLiuDing(Mat &img) {
    std::vector<float> fJump;
    int whiteCount = 0;
    const int x = 7;
    Mat jump = Mat::zeros(1, img.rows, CV_32F);
    for (int i = 0; i < img.rows; i++) {
      int jumpCount = 0;

      for (int j = 0; j < img.cols - 1; j++) {
        if (img.at<char>(i, j) != img.at<char>(i, j + 1)) jumpCount++;

        if (img.at<uchar>(i, j) == 255) {
          whiteCount++;
        }
      }

      jump.at<float>(i) = (float) jumpCount;
    }

    int iCount = 0;
    for (int i = 0; i < img.rows; i++) {
      fJump.push_back(jump.at<float>(i));
      if (jump.at<float>(i) >= 16 && jump.at<float>(i) <= 45) {

        // jump condition
        iCount++;
      }
    }

    // if not is not plate
    if (iCount * 1.0 / img.rows <= 0.40) {
      return false;
    }

    if (whiteCount * 1.0 / (img.rows * img.cols) < 0.15 ||
        whiteCount * 1.0 / (img.rows * img.cols) > 0.50) {
      return false;
    }

    for (int i = 0; i < img.rows; i++) {
      if (jump.at<float>(i) <= x) {
        for (int j = 0; j < img.cols; j++) {
          img.at<char>(i, j) = 0;
        }
      }
    }
    return true;
  }


void charsegment::clearBorder(const Mat &img, Rect& cropRect) {
  int r = img.rows;
  int c = img.cols;
  Mat boder = Mat::zeros(1, r, CV_8UC1);
  const int noJunpCount_thresh = int(0.15f * c);

  // if nojumpcount >
  for (int i = 0; i < r; i++) {
    int nojumpCount = 0;
    int isBorder = 0;
    for (int j = 0; j < c - 1; j++) {
      if (img.at<char>(i, j) == img.at<char>(i, j + 1))
        nojumpCount++;
      if (nojumpCount > noJunpCount_thresh) {
        nojumpCount = 0;
        isBorder = 1;
        break;
      }
    }
    boder.at<char>(i) = (char) isBorder;
  }

  const int mintop = int(0.1f * r);
  const int maxtop = int(0.9f * r);

  int minMatTop = 0;
  int maxMatTop = r - 1;

  for (int i = 0; i < mintop; i++) {
    if (boder.at<char>(i) == 1) {
      minMatTop = i;
    }
  }

  for (int i = r - 1; i > maxtop; i--) {
    if (boder.at<char>(i) == 1) {
      maxMatTop = i;
    }
  }

  cropRect = Rect(0, minMatTop, c, maxMatTop - minMatTop + 1);

}


Mat charsegment::preprocessChar(Mat in){
	//Remap image
	int h=in.rows;
	int w=in.cols;
	int charSize=20;	//统一每个字符的大小
	Mat transformMat=Mat::eye(2,3,CV_32F);
	int m=max(w,h);
	transformMat.at<float>(0,2)=m/2 - w/2;
	transformMat.at<float>(1,2)=m/2 - h/2;

	Mat warpImage(m,m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0) );

	Mat out;
	resize(warpImage, out, Size(charSize, charSize) );

	return out;
}
