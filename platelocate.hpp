#ifndef PLATEJUDGE_H
#define PLATEJUDGE_H

#include "plate.hpp"

using namespace std;
using namespace cv;

class platedetect
{
private:
    //via Sobel detect get the picture array
    vector<plate> SobelDetectPlate;
    //via Color detect get the picture array
    vector<plate> ColorDetectPlate;
    //高斯模糊参数
    int m_GaussianBlurSize = 5;
    //闭操作size大小
    int m_MorphSizeWidth = 7;
    int m_MorphSizeHeight = 3;
    static const int DEFAULT_GAUSSIANBLUR_SIZE = 5;
    static const int SOBEL_SCALE = 1;
    static const int SOBEL_DELTA = 0;
    static const int SOBEL_DDEPTH = CV_16S;
    static const int SOBEL_X_WEIGHT = 1;
    static const int SOBEL_Y_WEIGHT = 0;
    static const int DEFAULT_MORPH_SIZE_WIDTH = 17;  // 17
    static const int DEFAULT_MORPH_SIZE_HEIGHT = 3;
    const int PLATEWIDTH = 136;
    const int PLATEHEIGHT = 36;
    const int PLATETYPE = CV_8UC3;
    float m_error = 0.9;
    float m_aspect = 4;
    float m_verifyMin = 1;
    float m_verifyMax = 30;
protected:
    void RotatedRectConvetToMat(Mat src, vector<RotatedRect> &RotatedRectIn, vector<plate> &OutPlate, Color r = UNKNOW);
    bool VerifyPlateSizes(RotatedRect mr);
public:
    void SobelLocate(Mat src);
    void ColorLocate(Mat src, Color r);
    void BluePlusYellowLocate(Mat &src);
    inline vector<plate> GetSobelDetectPlate(){return SobelDetectPlate;}
    inline vector<plate> GetColorDetectPlate(){return ColorDetectPlate;}
    static Mat colorMatch(const Mat &src, Mat &match, const Color r, const bool adaptive_minsv);
};

#endif // PLATEJUDGE_H
