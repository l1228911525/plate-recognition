#ifndef CHARSEGMENT_H
#define CHARSEGMENT_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "plate.hpp"

using namespace std;
using namespace cv;

class charsegment
{
protected:
    void clearBorder(const Mat &img, Rect& cropRect);
    bool clearLiuDing(Mat &img);
    bool char_verifySizes(const RotatedRect & candidate);
    void char_sort(vector <RotatedRect > & in_char );
public:
    static Mat preprocessChar(Mat in);
    void PlateCharSegmentMat(const Mat & inputImg,vector <Mat>& dst_mat, Color r);
    void PlateCharSegmentPlate(plate &PlateIn, vector<Mat> &PlateChar);
    int CharSegmentPlate(Mat input, vector<Mat>& resultVec, Color color);
};

#endif // CHARSEGMENT_H
