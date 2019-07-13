#include "svm.hpp"
#include "svmfeature.hpp"

using namespace std;
using namespace cv;

void platesvm::PlateSvmTrain()
{

}

bool platesvm::PlateSvmJudgeMat(Mat &src)
{
    Mat feature;
    getHistomPlusColoFeatures(src, feature);
    feature = feature.reshape(1,1);
    float response = svm.predict(feature);
    if(response >= 0.5)
        return true;
    else
        return false;
}

bool platesvm::PlateSvmJudgePlate(plate &PlateIn)
{
    Mat src = PlateIn.GetPlateMat();
    return PlateSvmJudgeMat(src);
}

void platesvm::PlateSvmJudgeVecPlate(vector<plate> &VecPlateIn, vector<plate> &VecPlateOut)
{
    for(int i = 0; i < VecPlateIn.size(); i++)
    {
        Mat src = VecPlateIn[i].GetPlateMat();
        if(PlateSvmJudgeMat(src))
            VecPlateOut.push_back(VecPlateIn[i]);
    }
}

