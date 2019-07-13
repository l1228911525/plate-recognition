#ifndef SVM_H
#define SVM_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/ml/ml.hpp>
#include "plate.hpp"

using namespace cv;
using namespace std;

class platesvm
{
private:
    CvSVM svm;
public:
    platesvm(){}
    platesvm(char path[50]){svm.load(path);}
    void PlateSvmTrain();
    void SetPlateSvm(char path[50]){svm.load(path);}
    bool PlateSvmJudgeMat(Mat &src);
    bool PlateSvmJudgePlate(plate &PlateIn);
    void PlateSvmJudgeVecPlate(vector<plate> &VecPlateIn, vector<plate> &VecPlateOut);
};

#endif
