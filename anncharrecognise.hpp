#ifndef ANNCHARRECOGNISE_HPP
#define ANNCHARRECOGNISE_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <map>
#include "plate.hpp"

using namespace std;
using namespace cv;

class annrecognise
{
private:
    CvANN_MLP ann;
    int predictSize;
    string path;
    map<string, string> m_map;

public:
    annrecognise();
    void LoadModel();
    void LoadModel(string s);
    Mat ProjectedHistogram(Mat img, int t);
    Mat features(Mat in, int sizeData);
    int classify(Mat, bool);
    string charsIdentify(Mat input, bool isChinese);
    char classifyNumberPlusCharacterText(Mat);
};

#endif // ANNCHARRECOGNISE_HPP
