#include "platelocate.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;




void platedetect::SobelLocate(Mat src)
{
    vector<RotatedRect> rects;
    //保存高斯滤波的结果
    Mat src_blur, src_gray;
    Mat grad;

    int scale = SOBEL_SCALE;
    int delta = SOBEL_DELTA;
    int ddepth = SOBEL_DDEPTH;


    //高斯模糊。Size中的数字影响车牌定位的效果。
    GaussianBlur( src, src_blur, Size(m_GaussianBlurSize, m_GaussianBlurSize),
        0, 0, BORDER_DEFAULT );

    /// Convert it to gray
    cvtColor( src_blur, src_gray, CV_RGB2GRAY );
    equalizeHist( src_gray, src_gray);

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, SOBEL_Y_WEIGHT, 0, grad );
    //Laplacian( src_gray, grad_x, ddepth, 3, scale, delta, BORDER_DEFAULT );
    //convertScaleAbs( grad_x, grad );

    Mat img_threshold;
    threshold(grad, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
    //threshold(grad, img_threshold, 75, 255, CV_THRESH_BINARY);

    Mat element = getStructuringElement(MORPH_RECT, Size(m_MorphSizeWidth, m_MorphSizeHeight) );
    morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);

    //Find 轮廓 of possibles plates
    vector< vector< Point> > contours;
    findContours(img_threshold,
        contours, // a vector of contours
        CV_RETR_EXTERNAL, // 提取外部轮廓
        CV_CHAIN_APPROX_NONE); // all pixels of each contours
    Mat result;
    //Start to iterate to each contour founded
    vector<vector<Point> >::iterator itc = contours.begin();

    //Remove patch that are no inside limits of aspect ratio and area.
    int t = 0;
    while (itc != contours.end())
    {
        //Create bounding rect of object
        RotatedRect mr = minAreaRect(Mat(*itc));

        //large the rect for more
        if( !VerifyPlateSizes(mr))
        {
            itc = contours.erase(itc);
        }
        else
        {
            ++itc;
            rects.push_back(mr);
        }
    }
    RotatedRectConvetToMat(src, rects, SobelDetectPlate);
}

//via color feature locate the plate
void platedetect::ColorLocate(Mat src, Color r)
{
    vector<RotatedRect> rects;
    vector<vector<Point > > Contours;
    const int color_morph_width = 10;
    const int color_morph_height = 2;
    Mat Gray(src.size(), CV_8UC1);
    //!search for yellow, blue, white
    colorMatch(src, Gray, r, false);
    ///imshow("yellow", YellowGray);
    Mat element = getStructuringElement(
        MORPH_RECT, Size(color_morph_width, color_morph_height));

    //!close operation
    morphologyEx(Gray, Gray, MORPH_CLOSE, element);

    findContours(Gray,
               Contours,               // a vector of contours
               CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_NONE);

    vector<vector<Point > >::iterator itc = Contours.begin();
    while (itc != Contours.end())
    {
        RotatedRect mr = minAreaRect(Mat(*itc));
        if( !VerifyPlateSizes(mr))
        {
            itc = Contours.erase(itc);
        }
        else
        {
            ++itc;
            rects.push_back(mr);
        }
    }
    ///cout << rects.size() << endl;
    RotatedRectConvetToMat(src, rects, ColorDetectPlate, r);
}

//!color search, pixel of match color is became 255, not is became 0;
Mat platedetect::colorMatch(const Mat &src, Mat &match, const Color r,
    const bool adaptive_minsv)
{

    // if use adaptive_minsv
    // min value of s and v is adaptive to h
    const float max_sv = 255;
    const float minref_sv = 64;

    const float minabs_sv = 95; //95;threshold(yellowMat, yellowMat, 0, 255,


    // H range of blue

    const int min_blue = 100;  // 100
    const int max_blue = 140;  // 140

    // H range of yellow

    const int min_yellow = 15;  // 15
    const int max_yellow = 40;  // 40

    // H range of white

    const int min_white = 0;   // 15
    const int max_white = 30;  // 40

    Mat src_hsv;

    // convert to HSV space
    cvtColor(src, src_hsv, CV_BGR2HSV);

    std::vector<cv::Mat> hsvSplit;
    split(src_hsv, hsvSplit);
    equalizeHist(hsvSplit[2], hsvSplit[2]);
    merge(hsvSplit, src_hsv);

    // match to find the color

    int min_h = 0;
    int max_h = 0;
    switch (r) {
    case BLUE:
      min_h = min_blue;
      max_h = max_blue;
      break;
    case YELLOW:
      min_h = min_yellow;
      max_h = max_yellow;
      break;
    default:
      // Color::UNKNOWN
      break;
    }

    float diff_h = float((max_h - min_h) / 2);
    float avg_h = min_h + diff_h;

    int channels = src_hsv.channels();
    int nRows = src_hsv.rows;

    // consider multi channel image
    int nCols = src_hsv.cols * channels;
    if (src_hsv.isContinuous()) {
      nCols *= nRows;
      nRows = 1;
    }

    int i, j;
    uchar* p;
    float s_all = 0;
    float v_all = 0;
    float count = 0;
    for (i = 0; i < nRows; ++i) {
      p = src_hsv.ptr<uchar>(i);
      for (j = 0; j < nCols; j += 3) {
        int H = int(p[j]);      // 0-180
        int S = int(p[j + 1]);  // 0-255
        int V = int(p[j + 2]);  // 0-255

        s_all += S;
        v_all += V;
        count++;

        bool colorMatched = false;

        if (H > min_h && H < max_h) {
          float Hdiff = 0;
          if (H > avg_h)
            Hdiff = H - avg_h;
          else
            Hdiff = avg_h - H;

          float Hdiff_p = float(Hdiff) / diff_h;

          float min_sv = 0;
          if (true == adaptive_minsv)
            min_sv =
            minref_sv -
            minref_sv / 2 *
            (1
            - Hdiff_p);  // inref_sv - minref_sv / 2 * (1 - Hdiff_p)
          else
            min_sv = minabs_sv;  // add

          if ((S > min_sv && S < max_sv) && (V > min_sv && V < max_sv))
            colorMatched = true;
        }

        if (colorMatched == true) {
          p[j] = 0;
          p[j + 1] = 0;
          p[j + 2] = 255;
        }
        else {
          p[j] = 0;
          p[j + 1] = 0;
          p[j + 2] = 0;
        }
      }
    }

    // cout << "avg_s:" << s_all / count << endl;
    // cout << "avg_v:" << v_all / count << endl;

    // get the final binary

    Mat src_grey;
    std::vector<cv::Mat> hsvSplit_done;
    split(src_hsv, hsvSplit_done);
    src_grey = hsvSplit_done[2];
    match = src_grey;
    return src_grey;
}

void platedetect::RotatedRectConvetToMat(Mat src, vector<RotatedRect> &RotatedRectIn, vector<plate> &OutPlate, Color r)
{
    for(int i = 0; i < RotatedRectIn.size(); i++)
    {
        Mat rotmat = getRotationMatrix2D(RotatedRectIn[i].center, RotatedRectIn[i].angle, 1);
        Mat img_rotated;
        warpAffine(src, img_rotated, rotmat, src.size(), CV_INTER_CUBIC);
        Mat img_crop;
        getRectSubPix(img_rotated, RotatedRectIn[i].size, RotatedRectIn[i].center, img_crop);
        if(img_crop.rows > img_crop.cols)
        {
            transpose(img_crop, img_crop);
            flip(img_crop, img_crop, 0);
            //imshow("img_crop2", img_crop);
        }
        Mat result;
        result.create(PLATEHEIGHT, PLATEWIDTH, PLATETYPE);
        resize(img_crop, result, result.size(), 0, 0, INTER_CUBIC);
        plate CPlate(result, r, result.cols, result.rows, result.cols * result.rows);
        OutPlate.push_back(CPlate);
    }
}

bool platedetect::VerifyPlateSizes(RotatedRect mr)
{
    float error = m_error;
    // Spain car plate size: 52x11 aspect 4,7272
    // China car plate size: 440mm*140mm£¬aspect 3.142857

    // Real car plate size: 136 * 32, aspect 4
    float aspect = m_aspect;

    // Set a min and max area. All other patchs are discarded
    // int min= 1*aspect*1; // minimum area
    // int max= 2000*aspect*2000; // maximum area
    int min = 34 * 8 * m_verifyMin;  // minimum area
    int max = 34 * 8 * m_verifyMax;  // maximum area

    // Get only patchs that match to a respect ratio.
    float rmin = aspect - aspect * error;
    float rmax = aspect + aspect * error;

    float area = mr.size.height * mr.size.width;
    float r = (float) mr.size.width / (float) mr.size.height;
    if (r < 1) r = (float) mr.size.height / (float) mr.size.width;

    // cout << "area:" << area << endl;
    // cout << "r:" << r << endl;

    if ((area < min || area > max) || (r < rmin || r > rmax))
        return false;
    else
        return true;
}

void platedetect::BluePlusYellowLocate(Mat &src)
{
    ColorLocate(src, BLUE);
    ColorLocate(src, YELLOW);
}
